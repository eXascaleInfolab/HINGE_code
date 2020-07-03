import numpy as np
import random
import math
from itertools import repeat

def replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples):
    """
    Replace values randomly to get negative samples
    """
    rmd_dict = key_val #key2values

    new_range = (last_idx*num_negative_samples)


    for cur_idx in range(new_range): #loop batch size times
        key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2 #generate a number between (0 and arity-1) * 2. This is the column of a key, and we want to replace its value
        tmp_key = new_facts_indexes[last_idx + cur_idx, key_ind] #get the key. Row:last_idx+cur_idx, column:key_ind. In the first half of new_facts_indexes there are positive facts, and in the second half negative facts (that's why the row number is always last_idx (which is == batch size) + onther index)
        tmp_len = len(rmd_dict[tmp_key]) #get all the values for the key 'tmp_key'
        rdm_w = np.random.randint(0, tmp_len)  # [low,high) - Generate the index to get the key

        # Sample a random value
        times = 1 #loop counter
        tmp_array = new_facts_indexes[last_idx + cur_idx] #get the whole fact
        tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w] #replace the value
        while (tuple(tmp_array) in whole_train_facts): #if the new fact does not exist in the whole_train_facts (which contains also the permute facts)
            if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100): #if one of this condition is satisfied, then retun a fact with a random value
                tmp_array[key_ind+1] = np.random.randint(0, n_values)
            else: #else keep genrating a corrupted value
                rdm_w = np.random.randint(0, tmp_len)
                tmp_array[key_ind+1] = rmd_dict[tmp_key][rdm_w]
            times = times + 1
        new_facts_indexes[last_idx + cur_idx, key_ind+1] = tmp_array[key_ind+1] #store the corrupted value in new_facts_indexes (row:last_idx+cur_idx, column:key_ind+1)
        new_facts_values[last_idx + cur_idx] = [-1] #set -1 because we have generated a false fact

def replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, keyH2keyT, num_negative_samples):
    """
    Replace keys randomly to get negative samples
    """

    new_range = (last_idx*num_negative_samples)

    rdm_ws = np.random.randint(0, n_keys, new_range) #generate new_range random key ids. last_idx==batch_size

    for cur_idx in range(new_range): #loop batch size times
        key_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2 #generate a number between (0 and arity-1) * 2. This is the column of a key that we want to replace
        # Sample a random key
        tmp_array = new_facts_indexes[last_idx + cur_idx] #get a fact in position last_idx+cur_idx. In the first half of new_facts_indexes there are positive facts, and in the second half negative facts (that's why the row number is always last_idx (which is == batch size) + onther index)

        if key_ind==0 or key_ind==2: #replace column 0 and 2 with the same key
            tmp_array[0] = rdm_ws[cur_idx]
            tmp_array[2] = rdm_ws[cur_idx]
        else:
            tmp_array[key_ind] = rdm_ws[cur_idx]

        while (tuple(tmp_array) in whole_train_facts): # check if the corrupted fact exists in the whole_train_facts (which contains also the permute facts)

            rnd_key = np.random.randint(0, n_keys)
            if key_ind==0 or key_ind==2: #replace column 0 and 2 with the same key
                tmp_array[0] = rnd_key
                tmp_array[2] = rnd_key
            else:
                tmp_array[key_ind] = rnd_key

        new_facts_indexes[last_idx + cur_idx, key_ind] = tmp_array[key_ind] #store the corrupted key in new_facts_indexes (row:last_idx+cur_idx, column:key_ind)
        new_facts_values[last_idx + cur_idx] = [-1] #set -1 because we have generated a false fact

def Batch_Loader(train_i_indexes, train_i_values, n_values, n_keys, key_val, batch_size, arity, whole_train_facts, indexes_values, indexes_keys, keyH2keyT, num_negative_samples):
    # train_i_indexes: all facts with the same arity
    '''
    [[   0    0    1    1]
     [   2    0    3    2]
     [   2    0    3    3]
     ...
     [  48 1445   49 1446]
     [  48 1445   49 1440]
     [  48 1445   49 1442]]
    '''
    new_facts_indexes = np.empty((batch_size+(batch_size*num_negative_samples), 2*arity)).astype(np.int32) #matrix: batch_size+(batch_size*num_negative_samples), 2*arity
    new_facts_values = np.empty((batch_size+(batch_size*num_negative_samples), 1)).astype(np.float32) #matrix: batch_size+(batch_size*num_negative_samples), 1

    idxs = np.random.randint(0, len(train_i_values), batch_size) #create a list of batch_size random numbers between 0 and len(train_i_values)

    new_facts_indexes[:batch_size, :] = train_i_indexes[idxs, :] #train_i_indexes[idxs, :] takes the rows train_i_indexes with row number in idxs. train_i_indexes[idxs, :] is stored in the first batch_size rows of new_facts_indexes. The rows number > batch_size are not used in this line (but they will be used later).
    new_facts_values[:batch_size] = train_i_values[idxs, :] #put 1 in the first batch_size rows of new_facts_values
    last_idx = batch_size

    # Copy everyting in advance
    '''
    (E.g. batch size 3) new_facts_indexes BEFORE TILE:
    [[ 34 545  35 549]
     [ 48 918  49 930]
     [ 46 508  47 509]
     [  0   0   0   0]
     [  0   0   0   0]
     [  0   0   0   0]]
    '''
    new_facts_indexes[last_idx:last_idx+(last_idx*num_negative_samples), :] = np.tile(new_facts_indexes[:last_idx, :], (num_negative_samples, 1))
    '''
    new_facts_indexes AFTER TILE:
    [[ 34 545  35 549]
     [ 48 918  49 930]
     [ 46 508  47 509]
     [ 34 545  35 549]
     [ 48 918  49 930]
     [ 46 508  47 509]]
    '''

    #same here: we copy the first half new_facts_values in its second half.
    '''
    new_facts_values:
    [[ 1.]
     [ 1.]
     [ 1.]
     [ 0.]
     [ 0.]
     [ 0.]]
    '''
    new_facts_values[last_idx:last_idx+(last_idx*num_negative_samples)] = np.tile(new_facts_values[:last_idx], (num_negative_samples, 1))

    '''
    new_facts_values:
    [[1.]
     [1.]
     [1.]
     [1.]
     [1.]
     [1.]]
    '''
    val_key = random.uniform(0, 1) #fifty-fifty probability of generating negative samples for keys and values
    if val_key < 0.5:
        replace_val(n_values, last_idx, key_val, arity, new_facts_indexes, new_facts_values, whole_train_facts, num_negative_samples)
    else:
        replace_key(n_keys, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts, keyH2keyT, num_negative_samples)

    last_idx += batch_size

    return new_facts_indexes, new_facts_values
