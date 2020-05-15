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

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class HINGE5(torch.nn.Module):

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(HINGE5, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size-2), 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)


        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)


        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 3))
        zeros_(self.conv2.bias.data)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1)

        self.loss = torch.nn.Softplus()

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2role=None, id2value=None):
        fact_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,0:2]).flatten()).cuda(device)
        fact_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,0:2]).flatten()).cuda(device)

        if arity > 2:
            kv_roles_ids = torch.LongTensor(np.array(x_batch[:,0::2][:,2:]).flatten()).cuda(device)
            kv_values_ids = torch.LongTensor(np.array(x_batch[:,1::2][:,2:]).flatten()).cuda(device)
            kv_roles_embedded = self.emb_roles(kv_roles_ids).view(len(x_batch),arity-2,self.embedding_size)
            kv_values_embedded = self.emb_values(kv_values_ids).view(len(x_batch),arity-2,self.embedding_size)

        fact_roles_embedded = self.emb_roles(fact_roles_ids).view(len(x_batch),2,self.embedding_size)
        fact_values_embedded = self.emb_values(fact_values_ids).view(len(x_batch),2,self.embedding_size)

        fact_hrt_concat1 = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1).unsqueeze(1)
        fact_hrt_concat_vectors1 = self.conv1(fact_hrt_concat1)
        fact_hrt_concat_vectors1 = self.batchNorm1(fact_hrt_concat_vectors1)
        fact_hrt_concat_vectors1 = F.relu(fact_hrt_concat_vectors1).squeeze(3)

        fact_hrt_concat_vectors1 = fact_hrt_concat_vectors1.view(fact_hrt_concat_vectors1.size(0), -1).unsqueeze(2)

        if arity > 2:
            fact_hrt_concat3_hrt = torch.cat((fact_values_embedded[:,0,:].unsqueeze(1), fact_roles_embedded[:,0,:].unsqueeze(1), fact_values_embedded[:,1,:].unsqueeze(1)), 1)
            fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,0,:].unsqueeze(1), kv_values_embedded[:,0,:].unsqueeze(1)), 1).unsqueeze(1)
            fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
            fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp)
            fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
            fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
            fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)


            for i in range(arity-3):
                fact_hrt_concat3_tmp = torch.cat((fact_hrt_concat3_hrt, kv_roles_embedded[:,i+1,:].unsqueeze(1), kv_values_embedded[:,i+1,:].unsqueeze(1)), 1).unsqueeze(1)
                fact_hrt_concat_vectors3_tmp = self.conv2(fact_hrt_concat3_tmp)
                fact_hrt_concat_vectors3_tmp = self.batchNorm2(fact_hrt_concat_vectors3_tmp)
                fact_hrt_concat_vectors3_tmp = F.relu(fact_hrt_concat_vectors3_tmp).squeeze(3)
                fact_hrt_concat_vectors3_tmp = fact_hrt_concat_vectors3_tmp.view(fact_hrt_concat_vectors3_tmp.size(0), -1).unsqueeze(2)
                fact_hrt_concat_vectors1 = torch.cat((fact_hrt_concat_vectors1, fact_hrt_concat_vectors3_tmp), 2)

        min_val, _ = torch.min(fact_hrt_concat_vectors1, 2)

        evaluation_score = self.f_FCN_net(min_val)

        return evaluation_score
