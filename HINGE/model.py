import torch, math, itertools, os, psutil
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

def create_role_value_pairs (m_rel, arity):
    first = m_rel.repeat(1, arity, 1)
    second = m_rel.unsqueeze(2)
    second = second.repeat(1,1,arity,1).view(m_rel.size(0),-1,m_rel.size(2))
    concat_pair_of_rows = torch.cat((first,second), dim=2)
    return concat_pair_of_rows

def create_role_value_pairs_btw_ht_and_kv(fact_m_rel, kv_m_rel, rep_time):
    first = fact_m_rel.repeat(1, rep_time, 1)
    second = kv_m_rel.unsqueeze(2)
    second = second.repeat(1,1,2,1).view(kv_m_rel.size(0),-1,kv_m_rel.size(2))
    concat_pair_of_rows = torch.cat((first,second), dim=2)
    concat_pair_of_rows_2 = torch.cat((second,first), dim=2)
    concat_pair_of_rows_final = torch.cat((concat_pair_of_rows,concat_pair_of_rows_2), dim=1)
    return concat_pair_of_rows_final

def truncated_normal_(tensor, mean=0, std=1): #truncated normal distribution initializer
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class HINGE(torch.nn.Module): #different conv for hr/tr and kv

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters, 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded[:,1,:].size())
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)

        fact_hrt_concat2 = torch.cat((fact_values_embedded[:,1,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors2 = self.conv1(fact_hrt_concat2)
        fact_hrt_concat_vectors2 = self.batchNorm1(fact_hrt_concat_vectors2) #batch normalization
        fact_hrt_concat_vectors2 = F.relu(fact_hrt_concat_vectors2).squeeze(3)
        fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)


        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)

            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)



            # print("fact_hrt_concat_vectors3.size():", fact_hrt_concat_vectors3.size())
            # print("fact_hrt_concat_vectors3", fact_hrt_concat_vectors3)

        # fact_hrt_concat = fact_hrt_concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, 2, embsize*2]).
        # fact_hrt_concat_vectors = self.conv1(fact_hrt_concat) #fact_hrt_concat_vectors.size(): torch.Size([256, 200, 1, 100])
        # print("fact_hrt_concat_vectors.size():", fact_hrt_concat_vectors.size())
        # fact_hrt_concat_vectors = self.batchNorm1(fact_hrt_concat_vectors) #batch normalization
        # fact_hrt_concat_vectors = F.relu(fact_hrt_concat_vectors)
        # min_val, _ = torch.min(fact_hrt_concat_vectors, 1) #min_val.size(): torch.Size([256, 1, 100])
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())
        # print("fact_hrt_concat_vectors2.size():", fact_hrt_concat_vectors2.size())
        # fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)
        min_val, _ = torch.min(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # mean_val = torch.mean(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # min_val.squeeze()
        # print("min_val.size():", min_val.size())
        evaluation_score = self.f_FCN_net(min_val)

        return evaluation_score

class HINGE2(torch.nn.Module): #different conv for hr/tr and kv, but BCE loss

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE2, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters, 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.BCELoss()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded[:,1,:].size())
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)

        fact_hrt_concat2 = torch.cat((fact_values_embedded[:,1,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors2 = self.conv1(fact_hrt_concat2)
        fact_hrt_concat_vectors2 = self.batchNorm1(fact_hrt_concat_vectors2) #batch normalization
        fact_hrt_concat_vectors2 = F.relu(fact_hrt_concat_vectors2).squeeze(3)
        fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)


        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)

            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)



            # print("fact_hrt_concat_vectors3.size():", fact_hrt_concat_vectors3.size())
            # print("fact_hrt_concat_vectors3", fact_hrt_concat_vectors3)

        # fact_hrt_concat = fact_hrt_concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, 2, embsize*2]).
        # fact_hrt_concat_vectors = self.conv1(fact_hrt_concat) #fact_hrt_concat_vectors.size(): torch.Size([256, 200, 1, 100])
        # print("fact_hrt_concat_vectors.size():", fact_hrt_concat_vectors.size())
        # fact_hrt_concat_vectors = self.batchNorm1(fact_hrt_concat_vectors) #batch normalization
        # fact_hrt_concat_vectors = F.relu(fact_hrt_concat_vectors)
        # min_val, _ = torch.min(fact_hrt_concat_vectors, 1) #min_val.size(): torch.Size([256, 1, 100])
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())
        # print("fact_hrt_concat_vectors2.size():", fact_hrt_concat_vectors2.size())
        # fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)
        min_val, _ = torch.min(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # mean_val = torch.mean(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # min_val.squeeze()
        # print("min_val.size():", min_val.size())
        evaluation_score = self.f_FCN_net(min_val)
        evaluation_score = torch.sigmoid(evaluation_score)

        return evaluation_score

class HINGE3(torch.nn.Module): # same conv for tr, hr, kv...

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE3, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters, 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        # self.conv2 = torch.nn.Conv2d(1, num_filters, (5, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        # zeros_(self.conv2.bias.data) #custom initialization of bias
        # self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        # truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded[:,1,:].size())
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)

        fact_hrt_concat2 = torch.cat((fact_values_embedded[:,1,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors2 = self.conv1(fact_hrt_concat2)
        fact_hrt_concat_vectors2 = self.batchNorm1(fact_hrt_concat_vectors2) #batch normalization
        fact_hrt_concat_vectors2 = F.relu(fact_hrt_concat_vectors2).squeeze(3)
        fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)


        if arity > 2:
            fact_hrt_concat3_tmp = torch.cat((kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv1(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm1(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)

            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv1(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm1(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)



            # print("fact_hrt_concat_vectors3.size():", fact_hrt_concat_vectors3.size())
            # print("fact_hrt_concat_vectors3", fact_hrt_concat_vectors3)

        # fact_hrt_concat = fact_hrt_concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, 2, embsize*2]).
        # fact_hrt_concat_vectors = self.conv1(fact_hrt_concat) #fact_hrt_concat_vectors.size(): torch.Size([256, 200, 1, 100])
        # print("fact_hrt_concat_vectors.size():", fact_hrt_concat_vectors.size())
        # fact_hrt_concat_vectors = self.batchNorm1(fact_hrt_concat_vectors) #batch normalization
        # fact_hrt_concat_vectors = F.relu(fact_hrt_concat_vectors)
        # min_val, _ = torch.min(fact_hrt_concat_vectors, 1) #min_val.size(): torch.Size([256, 1, 100])
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())
        # print("fact_hrt_concat_vectors2.size():", fact_hrt_concat_vectors2.size())
        # fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)
        min_val, _ = torch.min(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # mean_val = torch.mean(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # min_val.squeeze()
        # print("min_val.size():", min_val.size())
        evaluation_score = self.f_FCN_net(min_val)

        return evaluation_score

class HINGE4(torch.nn.Module):

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE4, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size-2), 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.conv1_e = torch.nn.Conv2d(1, num_filters, (2, self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1_e.bias.data) #custom initialization of bias
        self.batchNorm1_e = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1_e.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN


        self.conv2 = torch.nn.Conv2d(1, num_filters, (3, num_filters)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        # if arity > 2: #if there are more than 2 key-value pairs in each fact
        #     kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
        #     kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
        #     kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
        #     kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded.size()) # fact_roles_embedded.size(): torch.Size([256, 2, 100])
        # print("fact_roles_embedded:", fact_roles_embedded)
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hr_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hr_concat_vectors1 = self.conv1(fact_hr_concat1)
        fact_hr_concat_vectors1 = self.batchNorm1(fact_hr_concat_vectors1) #batch normalization
        fact_hr_concat_vectors1 = F.relu(fact_hr_concat_vectors1).squeeze(3)
        # print("fact_hr_concat_vectors1.size():", fact_hr_concat_vectors1.size()) #fact_hr_concat_vectors1.size(): torch.Size([256, 200, 1, 98])

        fact_hr_concat_vectors1 = fact_hr_concat_vectors1.view(fact_hr_concat_vectors1.size(0), -1)
        evaluation_score = self.f_FCN_net(fact_hr_concat_vectors1)


        # fact_tr_concat2 = torch.cat((fact_values_embedded[:,1,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        # fact_tr_concat_vectors2 = self.conv1(fact_tr_concat2)
        # fact_tr_concat_vectors2 = self.batchNorm1(fact_tr_concat_vectors2) #batch normalization
        # fact_tr_concat_vectors2 = F.relu(fact_tr_concat_vectors2).squeeze(3)
        #
        # fact_ht_concat3 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        # # print("fact_ht_concat3.size():", fact_ht_concat3.size())
        # fact_ht_concat_vectors3 = self.conv1_e(fact_ht_concat3)
        # fact_ht_concat_vectors3 = self.batchNorm1_e(fact_ht_concat_vectors3) #batch normalization
        # fact_ht_concat_vectors3 = F.relu(fact_ht_concat_vectors3).squeeze(3)
        # # print("fact_ht_concat_vectors3.size():", fact_ht_concat_vectors3.size())
        #
        # fact_relatedness_vector = torch.cat((fact_hr_concat_vectors1, fact_tr_concat_vectors2, fact_ht_concat_vectors3), 2).unsqueeze(1)
        # # print("fact_relatedness_vector.size():", fact_relatedness_vector.size())
        # fact_relatedness = self.conv2(fact_relatedness_vector.permute(0,1,3,2))
        # fact_relatedness = self.batchNorm2(fact_relatedness) #batch normalization
        # fact_relatedness = F.relu(fact_relatedness).squeeze(3)
        # fact_relatedness = fact_relatedness.squeeze(2)
        # print("fact_relatedness.size():", fact_relatedness.size())
        # print("fact_relatedness:", fact_relatedness)




        # if arity > 2:
        #     fact_hrt_concat3_tmp = torch.cat((kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
        #     fact_hrt_concat_vectors3_tmp = self.conv1(fact_hrt_concat3_tmp)
        #     fact_hrt_concat_vectors3_tmp = self.batchNorm1(fact_hrt_concat_vectors3_tmp) #batch normalization
        #     fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
        #     fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)
        #
        #     for i in range(arity-3):
        #         fact_hrt_concat3_tmp = torch.cat((kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
        #         fact_hrt_concat_vectors3_tmp = self.conv1(fact_hrt_concat3_tmp)
        #         fact_hrt_concat_vectors3_tmp = self.batchNorm1(fact_hrt_concat_vectors3_tmp) #batch normalization
        #         fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
        #         fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors3, fact_hrt_concat_vectors3_tmp), 2)



            # print("fact_hrt_concat_vectors3.size():", fact_hrt_concat_vectors3.size())
            # print("fact_hrt_concat_vectors3", fact_hrt_concat_vectors3)

        # fact_hrt_concat = fact_hrt_concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, 2, embsize*2]).
        # fact_hrt_concat_vectors = self.conv1(fact_hrt_concat) #fact_hrt_concat_vectors.size(): torch.Size([256, 200, 1, 100])
        # print("fact_hrt_concat_vectors.size():", fact_hrt_concat_vectors.size())
        # fact_hrt_concat_vectors = self.batchNorm1(fact_hrt_concat_vectors) #batch normalization
        # fact_hrt_concat_vectors = F.relu(fact_hrt_concat_vectors)
        # min_val, _ = torch.min(fact_hrt_concat_vectors, 1) #min_val.size(): torch.Size([256, 1, 100])
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())
        # print("fact_hrt_concat_vectors2.size():", fact_hrt_concat_vectors2.size())
        # fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)
        # min_val, _ = torch.min(fact_htr_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # mean_val = torch.mean(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # min_val.squeeze()
        # print("min_val.size():", min_val.size())
        # evaluation_score = self.f_FCN_net(fact_relatedness.unsqueeze(1))

        return evaluation_score

class HINGE5(torch.nn.Module): # filter Size :,3

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE5, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size-2), 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN


        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 3)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded.size()) # fact_roles_embedded.size(): torch.Size([256, 2, 100])
        # print("fact_roles_embedded:", fact_roles_embedded)
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)
        # print("fact_hr_concat_vectors1.size():", fact_hr_concat_vectors1.size()) #fact_hr_concat_vectors1.size(): torch.Size([256, 200, 1, 98])

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
            # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size()) #fact_hrt_concat_vectors1.size(): torch.Size([256, 9800])
            # print("fact_hrt_concat_vectors3_tmp.size():", fact_hrt_concat_vectors3_tmp.size()) #fact_hrt_concat_vectors3_tmp.size(): torch.Size([256, 9800])
            fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
            # print("fact_hr_concat_vectors1.size():", fact_hrt_concat_vectors1.size())


            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
                fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
                # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        min_val, _ = torch.min(fact_hrt_concat_vectors1, 2) #min_val.size(): torch.Size([256, 1, 100])

        # print("min_val1.size():", min_val[:int(min_val.size(0)/2),:].size())
        # print("min_val2.size():", min_val[int(min_val.size(0)/2):,:].size())
        # temp = min_val[:int(min_val.size(0)/2),:]- min_val[int(min_val.size(0)/2):,:]
        # temp = temp.view(-1)
        # temp = (temp[temp > 0].size(0) - temp[temp < 0].size(0))/(min_val.size(0)/2)

        # temp = temp.mean() # does not really work


        # print("mean_diff.size():", temp)

        evaluation_score = self.f_FCN_net(min_val)

            # print("fact_hrt_concat_vectors3.size():", fact_hrt_concat_vectors3.size())
            # print("fact_hrt_concat_vectors3", fact_hrt_concat_vectors3)

        # fact_hrt_concat = fact_hrt_concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, 2, embsize*2]).
        # fact_hrt_concat_vectors = self.conv1(fact_hrt_concat) #fact_hrt_concat_vectors.size(): torch.Size([256, 200, 1, 100])
        # print("fact_hrt_concat_vectors.size():", fact_hrt_concat_vectors.size())
        # fact_hrt_concat_vectors = self.batchNorm1(fact_hrt_concat_vectors) #batch normalization
        # fact_hrt_concat_vectors = F.relu(fact_hrt_concat_vectors)
        # min_val, _ = torch.min(fact_hrt_concat_vectors, 1) #min_val.size(): torch.Size([256, 1, 100])
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())
        # print("fact_hrt_concat_vectors2.size():", fact_hrt_concat_vectors2.size())
        # fact_hrt_concat_vectors3 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors2), 2)
        # min_val, _ = torch.min(fact_htr_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # mean_val = torch.mean(fact_hrt_concat_vectors3, 2) #min_val.size(): torch.Size([256, 1, 100])
        # min_val.squeeze()
        # print("min_val.size():", min_val.size())
        # evaluation_score = self.f_FCN_net(fact_relatedness.unsqueeze(1))

        return evaluation_score

class HINGE6(torch.nn.Module): # filter Size :,1

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE6, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size), 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 1)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN


        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 1)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded.size()) # fact_roles_embedded.size(): torch.Size([256, 2, 100])
        # print("fact_roles_embedded:", fact_roles_embedded)
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)
        # print("fact_hr_concat_vectors1.size():", fact_hr_concat_vectors1.size()) #fact_hr_concat_vectors1.size(): torch.Size([256, 200, 1, 98])

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)
        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
            # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size()) #fact_hrt_concat_vectors1.size(): torch.Size([256, 9800])
            # print("fact_hrt_concat_vectors3_tmp.size():", fact_hrt_concat_vectors3_tmp.size()) #fact_hrt_concat_vectors3_tmp.size(): torch.Size([256, 9800])
            fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
            # print("fact_hr_concat_vectors1.size():", fact_hrt_concat_vectors1.size())


            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
                fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
                # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        min_val, _ = torch.min(fact_hrt_concat_vectors1, 2) #min_val.size(): torch.Size([256, 1, 100])

        evaluation_score = self.f_FCN_net(min_val)

        return evaluation_score

class HINGE7(torch.nn.Module): # hinge5+dropout

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE7, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        # self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        # xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        # zeros_(self.g_FCN_net.bias.data) #custom initialization of bias

        # self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size-2), 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias
        self.drop = torch.nn.Dropout(0.2)

        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        # self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN


        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 3)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv2.bias.data) #custom initialization of bias
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN

        self.loss = torch.nn.Softplus()
        # self.loss = torch.nn.MarginRankingLoss(margin=0.1)

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        # ############################################################################################################
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device) # get the role ids for the first two keys
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device) # get the value ids for the first two values

        if arity > 2: #if there are more than 2 key-value pairs in each fact
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device) # get the role ids of all roles except the first 2
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device) # get the value ids of all values except the first 2
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)
            # print("kv_roles_embedded.size():", kv_roles_embedded.size())
            # print("kv_values_embedded.size():", kv_values_embedded.size())

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 key ids (head and tail)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size) #2 because we want to get only the embeddings of first 2 value ids (values of head and tail)
        # print("fact_roles_embedded.size():", fact_roles_embedded.size()) # fact_roles_embedded.size(): torch.Size([256, 2, 100])
        # print("fact_roles_embedded:", fact_roles_embedded)
        # print("fact_values_embedded.size():", fact_values_embedded.size())

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1) #batch normalization
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)
        # print("fact_hr_concat_vectors1.size():", fact_hr_concat_vectors1.size()) #fact_hr_concat_vectors1.size(): torch.Size([256, 200, 1, 98])

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)

        # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
            # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size()) #fact_hrt_concat_vectors1.size(): torch.Size([256, 9800])
            # print("fact_hrt_concat_vectors3_tmp.size():", fact_hrt_concat_vectors3_tmp.size()) #fact_hrt_concat_vectors3_tmp.size(): torch.Size([256, 9800])
            fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
            # print("fact_hr_concat_vectors1.size():", fact_hrt_concat_vectors1.size())


            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp) #batch normalization
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
                fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)
                # print("fact_hrt_concat_vectors1.size():", fact_hrt_concat_vectors1.size())

        fact_hrt_concat_vectors1 = self.drop(fact_hrt_concat_vectors1)

        min_val, _ = torch.min(fact_hrt_concat_vectors1, 2) #min_val.size(): torch.Size([256, 1, 100])
        evaluation_score = self.f_FCN_net(min_val)

        return evaluation_score
