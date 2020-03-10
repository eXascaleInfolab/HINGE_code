import tensorflow as tf
import numpy as np
import pickle
import time
ISOTIMEFORMAT='%Y-%m-%d %X'

tf.compat.v1.flags.DEFINE_string("data_dir", "./data", "The data dir.")
tf.compat.v1.flags.DEFINE_string("bin_postfix", "", "The new_postfix for the output bin file.")
tf.compat.v1.flags.DEFINE_boolean("if_permutate", False, "If permutate for test filter.")



FLAGS = tf.compat.v1.flags.FLAGS
import sys
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

def permutations(arr, position, end, res):
    """
    Permutate the array
    """
    if position == end:
        res.append(tuple(arr))
    else:
        for index in range(position, end):
            arr[index], arr[position] = arr[position], arr[index]
            permutations(arr, position+1, end, res)
            arr[index], arr[position] = arr[position], arr[index]
    return res

def load_data_from_txt(filenames, values_indexes = None, roles_indexes = None, ary_permutation = None):
    """
    Take a list of file names and build the corresponding dictionnary of facts
    """
    if values_indexes is None:
        values_indexes= dict()
        values = set()
        next_val = 0
    else:
        values = set(values_indexes)
        next_val = max(values_indexes.values()) + 1

    if roles_indexes is None:
        roles_indexes= dict()
        roles= set()
        next_role = 0
    else:
        roles = set(roles_indexes)
        next_role = max(roles_indexes.values()) + 1
    if ary_permutation is None:
        ary_permutation= dict()

    max_n = 2  # The maximum arity of the facts
    for filename in filenames:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                xx_dict = eval(line)
                xx = xx_dict['N']
                if xx > max_n:
                    max_n = xx
    data = []
    for i in range(max_n-1):
        data.append(dict())

    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()

        for _, line in enumerate(lines):
            aline = ()
            xx_dict = eval(line)
            for k in xx_dict:
                if k == 'N':
                    continue
                if k in roles:
                    role_ind = roles_indexes[k]
                else:
                    role_ind = next_role
                    next_role += 1
                    roles_indexes[k] = role_ind
                    roles.add(k)
                if type(xx_dict[k]) == str:
                    val = xx_dict[k]
                    if val in values:
                        val_ind = values_indexes[val]
                    else:
                        val_ind = next_val
                        next_val += 1
                        values_indexes[val] = val_ind
                        values.add(val)
                    aline = aline + (role_ind,)
                    aline = aline + (val_ind,)
                else:
                    for val in xx_dict[k]:  # Multiple values
                        if val in values:
                            val_ind = values_indexes[val]
                        else:
                            val_ind = next_val
                            next_val += 1
                            values_indexes[val] = val_ind
                            values.add(val)
                        aline = aline + (role_ind,)
                        aline = aline + (val_ind,)

            if FLAGS.if_permutate == True:  # Permutate the elements in the fact for negative sampling or further computing the filtered metrics in the test process
                if xx_dict['N'] in ary_permutation:
                    res = ary_permutation[xx_dict['N']]
                else:
                    res = []
                    arr = np.array(range(xx_dict['N']))
                    res = permutations(arr, 0, len(arr), res)
                    ary_permutation[xx_dict['N']] = res
                for tpl in res:
                    tmpline = ()
                    for tmp_ind in tpl:
                        tmpline = tmpline + (aline[2*tmp_ind], aline[2*tmp_ind+1])
                    data[xx_dict['N']-2][tmpline] = [1]
            else:
                data[xx_dict['N']-2][aline] = [1]

    return data, values_indexes, roles_indexes, ary_permutation

def get_neg_candidate_set(folder, values_indexes, roles_indexes):
    """
    Get negative candidate set for replacing value
    """
    role_val = {}
    with open(folder + 'n-ary_train.json') as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        n_dict = eval(line)
        for k in n_dict:
            if k == 'N':
                continue
            k_ind = roles_indexes[k]
            if k_ind not in role_val:
                role_val[k_ind] = []
            v = n_dict[k]
            if type(v) == str:
                v_ind = values_indexes[v]
                if v_ind not in role_val[k_ind]:
                    role_val[k_ind].append(v_ind)
            else:  # Multiple values
                for val in v:
                    val_ind = values_indexes[val]
                    if val_ind not in role_val[k_ind]:
                        role_val[k_ind].append(val_ind)
    return role_val

def build_data(folder='data/'):
    """
    Build data and save to files
    """
    train_facts, values_indexes, roles_indexes, ary_permutation = load_data_from_txt([folder + 'n-ary_train.json'])
    valid_facts, values_indexes, roles_indexes, ary_permutation = load_data_from_txt([folder + 'n-ary_valid.json'],
            values_indexes = values_indexes , roles_indexes = roles_indexes, ary_permutation = ary_permutation)
    test_facts, values_indexes, roles_indexes, ary_permutation = load_data_from_txt([folder + 'n-ary_test.json'],
            values_indexes = values_indexes , roles_indexes = roles_indexes, ary_permutation = ary_permutation)
    data_info = {}
    data_info["train_facts"] = train_facts
    data_info["valid_facts"] = valid_facts
    data_info['test_facts'] = test_facts
    data_info['values_indexes'] = values_indexes
    data_info['roles_indexes'] = roles_indexes
    if FLAGS.if_permutate == False:
        role_val = get_neg_candidate_set(folder, values_indexes, roles_indexes)
        data_info['role_val'] = role_val
    with open(folder + "/dictionaries_and_facts" + FLAGS.bin_postfix + ".bin", 'wb') as f:
        pickle.dump(data_info, f)

if __name__ == '__main__':
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
    afolder = FLAGS.data_dir + '/'
    build_data(folder=afolder)
    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
