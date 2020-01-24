import torch, math, itertools, os, psutil
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from spodernet.utils.global_config import Config
from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

def create_role_value_pairs (m_rel, arity):
    first = m_rel.repeat(1, arity, 1)
    second = m_rel.unsqueeze(2)
    second = second.repeat(1,1,arity,1).view(m_rel.size(0),-1,m_rel.size(2))
    concat_pair_of_rows = torch.cat((first,second), dim=2)
    return concat_pair_of_rows

def truncated_normal_(tensor, mean=0, std=1): #truncated normal distribution initializer
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)



class NaLP(torch.nn.Module):

    def __init__(self, num_roles, num_values, embedding_size, num_filters=200, ngfcn=1200):
        super(NaLP, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.g_FCN_net = torch.nn.Linear((2*num_filters), ngfcn) #FCN: https://bit.ly/2Rx1C4I
        xavier_normal_(self.g_FCN_net.weight.data) #custom initialization of g_FCN_net
        zeros_(self.g_FCN_net.bias.data) #custom initialization of bias
        self.f_FCN_net = torch.nn.Linear(ngfcn, 1)
        xavier_normal_(self.f_FCN_net.weight.data) #custom initialization of f_FCN_net
        zeros_(self.f_FCN_net.bias.data) #custom initialization of bias
        self.emb_roles = torch.nn.Embedding(num_roles, self.embedding_size, padding_idx=0)
        self.emb_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (1, 2*self.embedding_size)) # https://bit.ly/33vdQQT in_channels (1 like black and white images), out_channels (number of future maps - set empirically), kernel_size (size of the filters), stride (filter convolves around the input volume by shifting one unit at a time), padding (zeros around the border)
        zeros_(self.conv1.bias.data) #custom initialization of bias
        self.batchNorm = torch.nn.BatchNorm2d(num_filters, momentum=0.1) #num_filters is the size of out_channels
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1) #custom initialization of filters in conv NN
        self.loss = torch.nn.Softplus()

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_roles.weight.data, -bound, bound)
        uniform_(self.emb_values.weight.data, -bound, bound)

    def forward_OLD(self, roles, values, batch_number, arity, debug, id2role, id2value, isTesting): #roles and values are tensors containing the indexes of the corresponding roles and values in the dictionary

        roles_embedded = self.emb_roles(roles).squeeze(2) #size FOR TRAINING: [batch_size*2, arity, 1, emb_size]. squeeze(2) removes the 3rd dimension. FOR EVALUATION: [num_of_replicas, arity, 1, emb_size]
        values_embedded = self.emb_values(values).squeeze(2) #size FOR TRAINING: [batch_size*2, arity, 1, emb_size]. squeeze(2) removes the 3rd dimension. FOR EVALUATION: [num_of_replicas, arity, 1, emb_size]

        if debug=='True':

            print("positive/negative roles ids:", roles)
            print("positive/negative values ids:", values)

            print("roles_embedded:", roles_embedded)
            print("values_embedded:", values_embedded)

            if isTesting:
                current_batch_size = roles_embedded.size()[0]
                number_of_replicas = roles_embedded.size()[0]
                current_arity = roles_embedded.size()[1]
                current_emb_size = roles_embedded.size()[2]
                print("current_batch_size:", current_batch_size)
                print("number_of_replicas:", number_of_replicas)
                print("current_arity:", current_arity)
                print("current_emb_size:", current_emb_size)
                print("roles_embedded.size():", roles_embedded.size()) #torch.Size([batch_size, number_of_replicas, 1, arity, emb_size])
                print("values_embedded.size():", values_embedded.size()) #torch.Size([batch_size, number_of_replicas, 1, arity, emb_size])


        ## ROLE-VALUE PAIR EMBEDDING
        #concatenation
        concat = torch.cat((roles_embedded, values_embedded), 2) #size FOR TRAINING: [batch_size*2, arity, emb_size*2]. FOR EVALUATION: [num_of_replicas, arity, emb_size*2]
        concat = concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, arity, embsize*2]). FOR EVALUATION: [num_of_replicas, 1, arity, emb_size*2]

        if debug=='True' and not isTesting:
            batch_size = roles.size(2)
            tmp_concat = torch.cat((roles[:batch_size], values[:batch_size]), 2) #concatenates positive role-value ids instead of the embeddings. NOTE that FOR TRAINING the size of roles and values is batch_size*2, where the first half are positives, while the second part are negatives.
            for row_it, tmp_fact in enumerate(tmp_concat):
                for role_value_pair in tmp_fact:
                    tmp_role = role_value_pair[0].item()
                    tmp_value = role_value_pair[1].item()
                    tmp_role = id2role[tmp_role]
                    tmp_value = id2value[tmp_value]
                    print("DEBUG 3", "row", (batch_number*batch_size)+row_it+1, "role-value pair:", tmp_role, tmp_value)
                print("")

        '''
        concat (3 facts, each with arity 2 (two k-v pairs). r-v are the embeddings of r and v concatenated)
        [[[[r0-v0 concatenated], [r1-v1 concatenated]]],
         [[[r0-v0 concatenated], [r1-v1 concatenated]]],
         [[[r0-v0 concatenated], [r1-v1 concatenated]]]]
        '''
        if debug=='True':
            print("concat:", concat)
            print("concat.size():", concat.size())

        future_vectors = self.conv1(concat) #FOR TRAINING: torch.Size([batch_size*2, num_filters, arity, 1]). FOR EVALUATION: torch.Size([num_of_replicas, num_filters, arity, 1])
        future_vectors = self.batchNorm(future_vectors) #batch normalization

        future_vectors_relued = F.relu(future_vectors)
        if debug=='True':
            print("future_vectors_relued.size():", future_vectors_relued.size())

        m_rel = future_vectors_relued.permute(0, 2, 1, 3).squeeze(3) #FOR TRAINING: torch.Size([batch_size*2, arity (m), num_filters, 1]). squeeze(3) removes the 4th dimension. FOR EVALUATION: torch.Size([num_of_replicas, arity (m), num_filters, 1]).
        if debug=='True':
            print("m_rel.size():", m_rel.size())

        ## RELATEDNESS EVALUATION
        # concatenate any row pairs of future_vectors
        concat_pair_of_rows = create_role_value_pairs(m_rel, arity) #FOR TRAINING: [batch_size*2, m^2, num_filters*2]. FOR EVALUATION: [num_of_replicas, m^2, num_filters*2]

        if debug=='True':
            print("concat_pair_of_rows:", concat_pair_of_rows)
            print("concat_pair_of_rows.size():", concat_pair_of_rows.size())

        #g-FCN
        g_FCN_out = self.g_FCN_net(concat_pair_of_rows) #FOR TRAINING: [batch_size*2, m^2, ngfcn]. FOR EVALUATION: [num_of_replicas, m^2, ngfcn]
        g_FCN_out = F.relu(g_FCN_out)
        if debug=='True':
            print("g_FCN_out.size():", g_FCN_out.size())

        #min
        min_val, _ = torch.min(g_FCN_out, 1) # FOR TRAINING: torch.Size([batch_size*2, ngfcn]). FOR EVALUATION: torch.Size([num_of_replicas, ngfcn]).
        if debug=='True':
            print("min_val.size():", min_val.size())


        #f-FCN
        evaluation_score = self.f_FCN_net(min_val) #FOR TRAINING: torch.Size([batch_size*2, 1]). FOR EVALUATION: torch.Size([num_of_replicas, 1])
        if debug=='True':
            print("evaluation_score.size():", evaluation_score.size())

        if isTesting:
            evaluation_score = torch.sigmoid(evaluation_score)

        return evaluation_score

    def forward(self, x_batch, arity, mode, device=None):

        # print("x_batch:", x_batch)

        '''
        x_batch: [[  20  785   21  429]
                 [  48 1402   49 1395]
                 [  20  373   21  377]
                 [  19  785   21  429]
                 [  52 1402   49 1395]
                 [ 100  373   21  377]]


        roles_embedded: tensor(
        [
            [[ vector role 20], [ vector role 21]],
            [[ vector role 48], [ vector role 49]],
            [[ vector role 20], [ vector role 21]],
            [[ vector role 19], [ vector role 21]],
            [[ vector role 52], [ vector role 49]],
            [[ vector role 100],[ vector role 21]]
        ], grad_fn=<StackBackward>)

        values_embedded: tensor([
            [[ vector value 785], [ vector value 429]],
            [[ vector value 1402],[ vector value 1395]],
            [[ vector value 373], [ vector value 377]],
            [[ vector value 785], [ vector value 429]],
            [[ vector value 1402],[ vector value 1395]],
            [[ vector value 373], [ vector value 377]]
        ], grad_fn=<StackBackward>)

        roles_embedded.size(): torch.Size([batch_size*2, arity, emb_size])
        values_embedded.size(): torch.Size([batch_size*2, arity, emb_size])

        '''

        

        roles_ids = torch.LongTensor(np.array(x_batch[:,0::2]).flatten()).cuda(device)
        values_ids = torch.LongTensor(np.array(x_batch[:,1::2]).flatten()).cuda(device)

        roles_embedded = self.emb_roles(roles_ids).view(len(x_batch),arity,self.embedding_size)
        values_embedded = self.emb_values(values_ids).view(len(x_batch),arity,self.embedding_size)

        # print("roles_embedded.size():", roles_embedded.size())
        # print("values_embedded.size():", values_embedded.size())

        ## ROLE-VALUE PAIR EMBEDDING
        #concatenation
        concat = torch.cat((roles_embedded, values_embedded), 2) #size FOR TRAINING: [batch_size*2, arity, emb_size*2]
        # print("concat.size() 1:", concat.size())
        concat = concat.unsqueeze(1) # reshaping 'concat' to 4D tensor (because conv1 needs a 4D tensor). We basically add one extra dimension at position 1. FOR TRAINING: torch.Size([batchsize*2, 1, arity, embsize*2]).
        # print("concat.size() 2:", concat.size())

        future_vectors = self.conv1(concat) #FOR TRAINING: torch.Size([batch_size*2, num_filters, arity, 1])
        # print("future_vectors.size() 1:", future_vectors.size())
        future_vectors = self.batchNorm(future_vectors) #batch normalization
        # print("future_vectors.size() 2:", future_vectors.size())
        future_vectors = F.relu(future_vectors)
        # print("future_vectors.size():", future_vectors.size())

        m_rel = future_vectors.permute(0, 2, 1, 3).squeeze(3) #FOR TRAINING: torch.Size([batch_size*2, arity (m), num_filters, 1]). squeeze(3) removes the 4th dimension
        # print("m_rel.size():", m_rel.size())

        ## RELATEDNESS EVALUATION
        # concatenate any row pairs of future_vectors
        concat_pair_of_rows = create_role_value_pairs(m_rel, arity) #FOR TRAINING: [batch_size*2, m^2, num_filters*2]
        # print("concat_pair_of_rows.size():", concat_pair_of_rows.size())

        #g-FCN
        g_FCN_out = self.g_FCN_net(concat_pair_of_rows) #FOR TRAINING: [batch_size*2, m^2, ngfcn]
        g_FCN_out = F.relu(g_FCN_out)
        # print("g_FCN_out.size() 2:", g_FCN_out.size())

        #min
        min_val, _ = torch.min(g_FCN_out, 1) # FOR TRAINING: torch.Size([batch_size*2, ngfcn])
        # print("min_val.size():", min_val.size())

        #f-FCN
        evaluation_score = self.f_FCN_net(min_val) #FOR TRAINING: torch.Size([batch_size*2, 1])
        # print("evaluation_score.size():", evaluation_score.size())

        return evaluation_score
