import argparse, sys, json, pickle, torch, operator, random, os
from model import *
import numpy as np
from random import randint
from random import randrange

from timeit import default_timer as timer
from datetime import timedelta
from random import shuffle

import more_itertools as mit
import torch.multiprocessing as mp

from copy import deepcopy
from ttictoc import TicToc

from batching import *

from multiprocessing import JoinableQueue, Queue, Process

def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i+n]

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus (model, epoch, n_roles, n_values, whole_train, whole_test, whole_valid, list_of_testing_facts, device, output_queue, new_eval_method, list_of_roles_h_ids, list_of_roles_t_ids, n_roles_h, dataset_without_h_and_t):
    with torch.no_grad():
        print("evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus on device", device)
        model.to(device)
        model.eval()

        range_of_roles = np.arange(n_roles)
        range_of_values = np.arange(n_values)

        binary_hit1_roles = 0
        nary_hit1_roles = 0
        overall_hit1_roles = 0
        binary_hit3_roles = 0
        nary_hit3_roles = 0
        overall_hit3_roles = 0
        binary_hit10_roles = 0
        nary_hit10_roles = 0
        overall_hit10_roles = 0
        binary_mrr_roles = 0
        nary_mrr_roles = 0
        overall_mrr_roles = 0
        binary_mrr_values = 0
        nary_mrr_values = 0
        binary_hit1_values = 0
        nary_hit1_values = 0
        binary_hit3_values = 0
        nary_hit3_values = 0
        binary_hit10_values = 0
        nary_hit10_values = 0
        overall_hit1_values = 0
        overall_hit3_values = 0
        overall_hit10_values = 0
        overall_mrr_values = 0

        number_of_binary_facts_roles = 0
        number_of_nary_facts_roles = 0
        number_of_total_facts_roles = 0
        number_of_binary_facts_values = 0
        number_of_nary_facts_values = 0
        number_of_total_facts_values = 0

        head_value_mrr = 0
        head_value_hit10 = 0
        head_value_hit3 = 0
        head_value_hit1 = 0
        tail_value_mrr = 0
        tail_value_hit10 = 0
        tail_value_hit3 = 0
        tail_value_hit1 = 0
        number_of_head_value_facts = 0
        number_of_tail_value_facts = 0
        head_key_mrr = 0
        head_key_hit10 = 0
        head_key_hit3 = 0
        head_key_hit1 = 0
        tail_key_mrr = 0
        tail_key_hit10 = 0
        tail_key_hit3 = 0
        tail_key_hit1 = 0
        number_of_head_key_facts = 0
        number_of_tail_key_facts = 0

        binary_head_key_mrr = 0
        binary_head_key_hit10 = 0
        binary_head_key_hit3 = 0
        binary_head_key_hit1 = 0
        number_of_binary_head_key_facts = 0
        nary_head_key_mrr = 0
        nary_head_key_hit10 = 0
        nary_head_key_hit3 = 0
        nary_head_key_hit1 = 0
        number_of_nary_head_key_facts = 0
        binary_tail_key_mrr = 0
        binary_tail_key_hit10 = 0
        binary_tail_key_hit3 = 0
        binary_tail_key_hit1 = 0
        number_of_binary_tail_key_facts = 0
        nary_tail_key_mrr = 0
        nary_tail_key_hit10 = 0
        nary_tail_key_hit3 = 0
        nary_tail_key_hit1 = 0
        number_of_nary_tail_key_facts = 0

        binary_head_value_mrr = 0
        binary_head_value_hit10 = 0
        binary_head_value_hit3 = 0
        binary_head_value_hit1 = 0
        number_of_binary_head_value_facts = 0
        nary_head_value_mrr = 0
        nary_head_value_hit10 = 0
        nary_head_value_hit3 = 0
        nary_head_value_hit1 = 0
        number_of_nary_head_value_facts = 0
        binary_tail_value_mrr = 0
        binary_tail_value_hit10 = 0
        binary_tail_value_hit3 = 0
        binary_tail_value_hit1 = 0
        number_of_binary_tail_value_facts = 0
        nary_tail_value_mrr = 0
        nary_tail_value_hit10 = 0
        nary_tail_value_hit3 = 0
        nary_tail_value_hit1 = 0
        number_of_nary_tail_value_facts = 0

        keys_without_hrt_mrr = 0 #without hrt
        keys_without_hrt_hit10 = 0 #without hrt
        keys_without_hrt_hit3 = 0 #without hrt
        keys_without_hrt_hit1 = 0 #without hrt
        number_keys_without_hrt = 0 #without hrt
        values_without_hrt_mrr = 0 #without hrt
        values_without_hrt_hit10 = 0 #without hrt
        values_without_hrt_hit3 = 0 #without hrt
        values_without_hrt_hit1 = 0 #without hrt
        number_values_without_hrt = 0 #without hrt


        for fact_progress, fact in enumerate(list_of_testing_facts):
            fact = list(fact)
            arity = int(len(fact)/2)


            #parse the fact by column and tile it
            for column in range (len(fact)):

                correct_index = fact[column]

                if column % 2 == 0: #roles
                    if new_eval_method == 'True':
                        if dataset_without_h_and_t == 'True':
                            if column==0 or column==2:
                                tiled_fact = np.array(fact*n_roles).reshape(n_roles,-1)
                                tiled_fact[:,0] = range_of_roles
                                tiled_fact[:,2] = range_of_roles
                            else:
                                tiled_fact = np.array(fact*n_roles).reshape(n_roles,-1)
                                tiled_fact[:,column] = range_of_roles
                        else:
                            if column==0 or column==2:
                                tiled_fact = np.array(fact*n_roles_h).reshape(n_roles_h,-1)
                                tiled_fact[:,0] = list_of_roles_h_ids
                                tiled_fact[:,2] = list_of_roles_t_ids
                            else:
                                tiled_fact = np.array(fact*n_roles).reshape(n_roles,-1)
                                tiled_fact[:,column] = range_of_roles
                    else:
                        tiled_fact = np.array(fact*n_roles).reshape(n_roles,-1)
                        tiled_fact[:,column] = range_of_roles

                    if arity == 2: number_of_binary_facts_roles = number_of_binary_facts_roles + 1
                    else: number_of_nary_facts_roles = number_of_nary_facts_roles + 1

                    if arity==2 and column==0:
                        number_of_binary_head_key_facts += 1
                    elif arity==2 and column==2:
                        number_of_binary_tail_key_facts += 1
                    elif arity>2 and column==0:
                        number_of_nary_head_key_facts += 1
                    elif arity>2 and column==2:
                        number_of_nary_tail_key_facts += 1

                    if column > 3:
                        number_keys_without_hrt += 1

                else:
                    tiled_fact = np.array(fact*n_values).reshape(n_values,-1)
                    tiled_fact[:,column] = range_of_values
                    if arity == 2: number_of_binary_facts_values = number_of_binary_facts_values + 1
                    else: number_of_nary_facts_values = number_of_nary_facts_values + 1

                    if arity==2 and column == 1:
                        number_of_binary_head_value_facts += 1
                    elif arity>2 and column == 1:
                        number_of_nary_head_value_facts += 1
                    elif arity==2 and column == 3:
                        number_of_binary_tail_value_facts += 1
                    elif arity>2 and column == 3:
                        number_of_nary_tail_value_facts += 1

                    if column > 3:
                        number_values_without_hrt += 1


                if column == 1:
                    number_of_head_value_facts += 1
                elif column == 3:
                    number_of_tail_value_facts += 1
                elif column == 0:
                    number_of_head_key_facts += 1
                elif column == 2:
                    number_of_tail_key_facts += 1


                tiled_fact = list(chunks(tiled_fact, 128))
                pred = model(tiled_fact[0], arity, "testing", device)
                for batch_it in range(1, len(tiled_fact)):
                    pred_tmp = model(tiled_fact[batch_it], arity, "testing", device)
                    pred = torch.cat((pred, pred_tmp))

                sorted_pred = torch.argsort(pred, dim=0, descending=True)
                if new_eval_method == 'True' and dataset_without_h_and_t == 'False':
                    if column==0:
                        sorted_pred = np.asarray(list_of_roles_h_ids)[sorted_pred.cpu().numpy()]
                    elif column==2:
                        sorted_pred = np.asarray(list_of_roles_t_ids)[sorted_pred.cpu().numpy()]

                position_of_correct_fact_in_sorted_pred = 0
                for tmpxx in sorted_pred:
                    if tmpxx == correct_index:
                        break
                    tmp_list = deepcopy(fact)
                    tmp_list[column] = tmpxx.item()
                    tmpTriple = tuple(tmp_list)
                    if (len(whole_train) > arity-2) and (tmpTriple in whole_train[arity-2]): # 2-ary in index 0
                        continue
                    elif (len(whole_valid) > arity-2) and (tmpTriple in whole_valid[arity-2]): # 2-ary in index 0
                        continue
                    elif (len(whole_test) > arity-2) and (tmpTriple in whole_test[arity-2]): # 2-ary in index 0
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred += 1

                if position_of_correct_fact_in_sorted_pred == 0:
                    if column % 2 == 0: #roles
                        overall_hit1_roles = overall_hit1_roles + 1
                        if arity == 2: #binary fact
                            binary_hit1_roles = binary_hit1_roles + 1
                        else: #nary fact
                            nary_hit1_roles = nary_hit1_roles + 1
                        if column > 3:
                            keys_without_hrt_hit1 += 1 #without hrt
                    else: #values
                        overall_hit1_values = overall_hit1_values + 1
                        if arity == 2: #binary fact
                            binary_hit1_values = binary_hit1_values + 1
                        else: #nary fact
                            nary_hit1_values = nary_hit1_values + 1
                        if column > 3:
                            values_without_hrt_hit1 += 1 #without hrt
                    if column == 1:
                        head_value_hit1 += 1
                        if arity == 2:
                            binary_head_value_hit1 += 1
                        else:
                            nary_head_value_hit1 += 1
                    elif column == 3:
                        tail_value_hit1 += 1
                        if arity == 2:
                            binary_tail_value_hit1 += 1
                        else:
                            nary_tail_value_hit1 += 1
                    elif column == 0:
                        head_key_hit1 += 1
                        if arity == 2:
                            binary_head_key_hit1 += 1
                        else:
                            nary_head_key_hit1 += 1
                    elif column == 2:
                        tail_key_hit1 += 1
                        if arity == 2:
                            binary_tail_key_hit1 += 1
                        else:
                            nary_tail_key_hit1 += 1

                if position_of_correct_fact_in_sorted_pred < 3:
                    if column % 2 == 0: #roles
                        overall_hit3_roles = overall_hit3_roles + 1
                        if arity == 2: #binary fact
                            binary_hit3_roles = binary_hit3_roles + 1
                        else: #nary fact
                            nary_hit3_roles = nary_hit3_roles + 1
                        if column > 3:
                            keys_without_hrt_hit3 += 1 #without hrt
                    else: #values
                        overall_hit3_values = overall_hit3_values + 1
                        if arity == 2: #binary fact
                            binary_hit3_values = binary_hit3_values + 1
                        else: #nary fact
                            nary_hit3_values = nary_hit3_values + 1
                        if column > 3:
                            values_without_hrt_hit3 += 1 #without hrt

                    if column == 1:
                        head_value_hit3 += 1
                        if arity == 2:
                            binary_head_value_hit3 += 1
                        else:
                            nary_head_value_hit3 += 1
                    elif column == 3:
                        tail_value_hit3 += 1
                        if arity == 2:
                            binary_tail_value_hit3 += 1
                        else:
                            nary_tail_value_hit3 += 1
                    elif column == 0:
                        head_key_hit3 += 1
                        if arity == 2:
                            binary_head_key_hit3 += 1
                        else:
                            nary_head_key_hit3 += 1
                    elif column == 2:
                        tail_key_hit3 += 1
                        if arity == 2:
                            binary_tail_key_hit3 += 1
                        else:
                            nary_tail_key_hit3 += 1

                if position_of_correct_fact_in_sorted_pred < 10:
                    if column % 2 == 0: #roles
                        overall_hit10_roles = overall_hit10_roles + 1
                        if arity == 2: #binary fact
                            binary_hit10_roles = binary_hit10_roles + 1
                        else: #nary fact
                            nary_hit10_roles = nary_hit10_roles + 1
                        if column > 3:
                            keys_without_hrt_hit10 += 1 #without hrt
                    else: #values
                        overall_hit10_values = overall_hit10_values + 1
                        if arity == 2: #binary fact
                            binary_hit10_values = binary_hit10_values + 1
                        else: #nary fact
                            nary_hit10_values = nary_hit10_values + 1
                        if column > 3:
                            values_without_hrt_hit10 += 1 #without hrt
                    if column == 1:
                        head_value_hit10 += 1
                        if arity == 2:
                            binary_head_value_hit10 += 1
                        else:
                            nary_head_value_hit10 += 1
                    elif column == 3:
                        tail_value_hit10 += 1
                        if arity == 2:
                            binary_tail_value_hit10 += 1
                        else:
                            nary_tail_value_hit10 += 1
                    elif column == 0:
                        head_key_hit10 += 1
                        if arity == 2:
                            binary_head_key_hit10 += 1
                        else:
                            nary_head_key_hit10 += 1
                    elif column == 2:
                        tail_key_hit10 += 1
                        if arity == 2:
                            binary_tail_key_hit10 += 1
                        else:
                            nary_tail_key_hit10 += 1

                if column % 2 == 0: #roles
                    overall_mrr_roles = overall_mrr_roles + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if arity == 2: #binary fact
                        binary_mrr_roles = binary_mrr_roles + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    else: #nary fact
                        nary_mrr_roles = nary_mrr_roles + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if column > 3: #without hrt
                        keys_without_hrt_mrr = keys_without_hrt_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                else: #values
                    overall_mrr_values = overall_mrr_values + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if arity == 2: #binary fact
                        binary_mrr_values = binary_mrr_values + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    else: #nary fact
                        nary_mrr_values = nary_mrr_values + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if column > 3: #without hrt
                        values_without_hrt_mrr = values_without_hrt_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1)) #+1 because otherwise if the predicted element is in top, it is going to divide by 0
                if column == 1:
                    head_value_mrr = head_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    if arity == 2:
                        binary_head_value_mrr = binary_head_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    else:
                        nary_head_value_mrr = nary_head_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                elif column == 3:
                    tail_value_mrr = tail_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    if arity == 2:
                        binary_tail_value_mrr = binary_tail_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    else:
                        nary_tail_value_mrr = nary_tail_value_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                elif column == 0:
                    head_key_mrr = head_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    if arity == 2:
                        binary_head_key_mrr = binary_head_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    else:
                        nary_head_key_mrr = nary_head_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                elif column == 2:
                    tail_key_mrr = tail_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    if arity == 2:
                        binary_tail_key_mrr = binary_tail_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))
                    else:
                        nary_tail_key_mrr = nary_tail_key_mrr + float(1/(position_of_correct_fact_in_sorted_pred+1))


    output_message = {}

    number_of_total_facts_roles = number_of_binary_facts_roles + number_of_nary_facts_roles
    if number_of_total_facts_roles > 0:
        output_message["overall_roles"] = [device, number_of_total_facts_roles, overall_mrr_roles, overall_hit10_roles, overall_hit3_roles, overall_hit1_roles]

    number_of_total_facts_values = number_of_binary_facts_values + number_of_nary_facts_values
    if number_of_total_facts_values > 0:
        output_message["overall_values"] = [device, number_of_total_facts_values, overall_mrr_values, overall_hit10_values, overall_hit3_values, overall_hit1_values]

    if number_of_binary_facts_roles > 0:
        output_message["binary_roles"] = [device, number_of_binary_facts_roles, binary_mrr_roles, binary_hit10_roles, binary_hit3_roles, binary_hit1_roles]

    if number_of_binary_facts_values > 0:
        output_message["binary_values"] = [device, number_of_binary_facts_values, binary_mrr_values, binary_hit10_values, binary_hit3_values, binary_hit1_values]

    if number_of_nary_facts_roles > 0:
        output_message["nary_roles"] = [device, number_of_nary_facts_roles, nary_mrr_roles, nary_hit10_roles, nary_hit3_roles, nary_hit1_roles]

    if number_of_nary_facts_values > 0:
        output_message["nary_values"] = [device, number_of_nary_facts_values, nary_mrr_values, nary_hit10_values, nary_hit3_values, nary_hit1_values]

    if number_of_head_value_facts > 0:
        output_message["head_value_facts"] = [device, number_of_head_value_facts, head_value_mrr, head_value_hit10, head_value_hit3, head_value_hit1]

    if number_of_tail_value_facts > 0:
        output_message["tail_value_facts"] = [device, number_of_tail_value_facts, tail_value_mrr, tail_value_hit10, tail_value_hit3, tail_value_hit1]

    if number_of_head_key_facts > 0:
        output_message["head_key_facts"] = [device, number_of_head_key_facts, head_key_mrr, head_key_hit10, head_key_hit3, head_key_hit1]

    if number_of_tail_key_facts > 0:
        output_message["tail_key_facts"] = [device, number_of_tail_key_facts, tail_key_mrr, tail_key_hit10, tail_key_hit3, tail_key_hit1]

    if number_of_binary_head_key_facts > 0:
        output_message["binary_head_key_facts"] = [device, number_of_binary_head_key_facts, binary_head_key_mrr, binary_head_key_hit10, binary_head_key_hit3, binary_head_key_hit1]

    if number_of_nary_head_key_facts > 0:
        output_message["nary_head_key_facts"] = [device, number_of_nary_head_key_facts, nary_head_key_mrr, nary_head_key_hit10, nary_head_key_hit3, nary_head_key_hit1]

    if number_of_binary_tail_key_facts > 0:
        output_message["binary_tail_key_facts"] = [device, number_of_binary_tail_key_facts, binary_tail_key_mrr, binary_tail_key_hit10, binary_tail_key_hit3, binary_tail_key_hit1]

    if number_of_nary_tail_key_facts > 0:
        output_message["nary_tail_key_facts"] = [device, number_of_nary_tail_key_facts, nary_tail_key_mrr, nary_tail_key_hit10, nary_tail_key_hit3, nary_tail_key_hit1]

    if number_of_binary_head_value_facts > 0:
          output_message["binary_head_value_facts"] = [device, number_of_binary_head_value_facts, binary_head_value_mrr, binary_head_value_hit10, binary_head_value_hit3, binary_head_value_hit1]

    if number_of_nary_head_value_facts > 0:
        output_message["nary_head_value_facts"] = [device, number_of_nary_head_value_facts, nary_head_value_mrr, nary_head_value_hit10, nary_head_value_hit3, nary_head_value_hit1]

    if number_of_binary_tail_value_facts > 0:
        output_message["binary_tail_value_facts"] = [device, number_of_binary_tail_value_facts, binary_tail_value_mrr, binary_tail_value_hit10, binary_tail_value_hit3, binary_tail_value_hit1]

    if number_of_nary_tail_value_facts > 0:
        output_message["nary_tail_value_facts"] = [device, number_of_nary_tail_value_facts, nary_tail_value_mrr, nary_tail_value_hit10, nary_tail_value_hit3, nary_tail_value_hit1]

    if number_keys_without_hrt > 0: #without hrt
        output_message["keys_without_hrt"] = [device, number_keys_without_hrt, keys_without_hrt_mrr, keys_without_hrt_hit10, keys_without_hrt_hit3, keys_without_hrt_hit1]

    if number_values_without_hrt > 0: #without hrt
        output_message["values_without_hrt"] = [device, number_values_without_hrt, values_without_hrt_mrr, values_without_hrt_hit10, values_without_hrt_hit3, values_without_hrt_hit1]


    output_queue.put(output_message)

    return output_queue

def prepare_data_for_evaluation_and_evaluate_on_multiple_gpus (model, test, epoch, n_roles, n_values, whole_train, whole_test, whole_valid, gpu_ids_splitted, output_queue, id2role, id2value, new_eval_method, list_of_roles_h_ids, list_of_roles_t_ids, n_roles_h, dataset_without_h_and_t):
    print("prepare_data_for_evaluation_and_evaluate_on_multiple_gpus")
    list_of_all_test_facts = []
    for test_fact_grouped_by_arity in test:
        for test_fact in test_fact_grouped_by_arity:
            list_of_all_test_facts.append(test_fact)

    shuffle(list_of_all_test_facts)
    slices = list(split_list(list_of_all_test_facts, len(gpu_ids_splitted)))

    jobs = []

    for slice_it, slice in enumerate(slices):
        device = "cuda:" + str(gpu_ids_splitted[slice_it])
        current_job = mp.Process(target=evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus, args=(model, epoch, n_roles, n_values, whole_train, whole_test, whole_valid, slices[slice_it], device, output_queue, new_eval_method, list_of_roles_h_ids, list_of_roles_t_ids, n_roles_h, dataset_without_h_and_t))
        jobs.append(current_job)

    # start all job
    for current_job in jobs:
        current_job.start()

    # exit the completed processes
    for current_job in jobs:
        current_job.join()

    results = [output_queue.get() for current_job in jobs]
    weighted_scores = {}
    for dictionary in results:
        for task in dictionary:
            if task not in weighted_scores:
                weighted_scores[task] = []
                # dictionary[task][0] is device
                weighted_scores[task].append(dictionary[task][1]) #number of facts
                weighted_scores[task].append(dictionary[task][2]) #mrr
                weighted_scores[task].append(dictionary[task][3]) #hits@10
                weighted_scores[task].append(dictionary[task][4]) #hits@3
                weighted_scores[task].append(dictionary[task][5]) #hits@1
            else:
                weighted_scores[task][0] = weighted_scores[task][0] + dictionary[task][1] #number of facts
                weighted_scores[task][1] = weighted_scores[task][1] + dictionary[task][2] #mrr
                weighted_scores[task][2] = weighted_scores[task][2] + dictionary[task][3] #hits@10
                weighted_scores[task][3] = weighted_scores[task][3] + dictionary[task][4] #hits@3
                weighted_scores[task][4] = weighted_scores[task][4] + dictionary[task][5] #hits@1

    for task in weighted_scores:
        tot_facts = weighted_scores[task][0]
        mrr = weighted_scores[task][1]
        hits10 = weighted_scores[task][2]
        hits3 = weighted_scores[task][3]
        hits1 = weighted_scores[task][4]
        print("")
        print("PAOLO final", task, "@ epoch", epoch, "mrr:", str(mrr) + "/" + str(tot_facts), "=", (mrr/tot_facts)*100, "%")
        print("PAOLO final", task, "@ epoch", epoch, "hits10:", str(hits10) + "/" + str(tot_facts), "=", (hits10/tot_facts)*100, "%")
        print("PAOLO final", task, "@ epoch", epoch, "hits3:", str(hits3) + "/" + str(tot_facts), "=", (hits3/tot_facts)*100, "%")
        print("PAOLO final", task, "@ epoch", epoch, "hits1:", str(hits1) + "/" + str(tot_facts), "=", (hits1/tot_facts)*100, "%")
        print("")

    ### END OF PARALLELIZATION ###
    print("Evaluation is over.")

def main():

    #parse input arguments
    parser = argparse.ArgumentParser(description="Model's hyperparameters")
    parser.add_argument('--indir', type=str, help='Input dir of train, test and valid data')
    parser.add_argument('--model', default="NaLP", type=str, help='Model (NaLP)')
    parser.add_argument('--epochs', default=10, help='Number of epochs (default: 10)' )
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)' )
    parser.add_argument('--ngfcn', type=int, default=1200, help='g_fcn size' )
    parser.add_argument('--num_filters', type=int, default=200, help='number of filters CNN' )
    parser.add_argument('--embsize', default=100, help='Embedding size (default: 100)' )
    parser.add_argument('--learningrate', default=0.00005, help='Learning rate (default: 0.00005)' )
    parser.add_argument('--debug', default='False', help='If true, it prints all the facts (default: False)' )
    parser.add_argument('--outdir', type=str, help='Output dir of model')
    parser.add_argument('--load', default='False', help='If true, it loads a saved model in dir outdir and evaluate it (default: False)' )
    parser.add_argument('--gpu_ids', default='0,1,2,3', help='Comma-separated gpu id used to paralellize the evaluation (default: 0,1,2,3)' )
    parser.add_argument('--new_eval_method', default='False', help='If true, it executes the new evaluation method' )
    parser.add_argument('--new_negative_sampling_h_and_t', default='False', help='If true, it executes the new negative sampling' )
    parser.add_argument('--new_batching_method', default='False', help='If true, it executes the new batching schema' )
    parser.add_argument('--num_negative_samples', type=int, default=1, help='number of negative samples for each positive sample' )
    parser.add_argument('--fifty_percent_prob_of_creating_neg_samples', default='False', help='If true, it generates with a 50-50 probability the negatives for keys and roles. If false, it uses the probability of NaLP where most of the negatives are values. ' )
    parser.add_argument('--dataset_without_h_and_t', default='False', help='Set True if you are using a dataset without _h and _t' )
    args = parser.parse_args()
    print("\n\n************************")
    for e in vars(args):
        print (e, getattr(args, e))
    print("************************\n\n")

    gpu_ids_splitted = list(map(int, args.gpu_ids.split(",")))

    if args.load == 'True':
        t2 = TicToc()
        print("Loading and evaluating model in", args.outdir)
        mp.set_start_method('spawn')
        with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
            data_info = pickle.load(fin)
        test = data_info["test_facts"]
        roles2id = data_info['roles_indexes'] #roles_indexes
        values2id = data_info['values_indexes'] #values_indexes
        n_roles = len(roles2id)
        n_values = len(values2id)

        roles_h_2id = {}
        roles_t_2id = {}
        id2roles_h = {}
        id2roles_t = {}
        roleH2roleT = {}
        if 'wikipeople' in args.indir.lower():
            print("\n********************************************************\ncreating extra dictionaries for wikipeople dataset\n********************************************************\n")
            for r in roles2id:
                if r.endswith("_h"):
                    roles_h_2id[r] = roles2id[r]
                    id2roles_h[roles2id[r]] = r
                elif r.endswith("_t"):
                    roles_t_2id[r] = roles2id[r]
                    id2roles_t[roles2id[r]] = r
            for r_h_id in id2roles_h:
                r_h_string = id2roles_h[r_h_id]
                r_t_string = r_h_string.replace("_h", "_t")
                r_t_id = roles_t_2id[r_t_string]
                roleH2roleT[r_h_id] = r_t_id
        elif 'jf17k' in args.indir.lower():
            print("\n********************************************************\ncreating extra dictionaries for JF17K dataset\n********************************************************\n")
            for r in roles2id:
                if r.endswith("0"):
                    roles_h_2id[r] = roles2id[r]
                    id2roles_h[roles2id[r]] = r
                elif r.endswith("1"):
                    roles_t_2id[r] = roles2id[r]
                    id2roles_t[roles2id[r]] = r
            for r_h_id in id2roles_h:
                r_h_string = id2roles_h[r_h_id]
                r_t_string = r_h_string[:-1] + '1'
                r_t_id = roles_t_2id[r_t_string]
                roleH2roleT[r_h_id] = r_t_id

        n_roles_h = len(roles_h_2id)
        n_roles_t = len(roles_t_2id)
        list_of_roles_h_ids = list(id2roles_h.keys())
        list_of_roles_t_ids = []
        for r_h_id in list_of_roles_h_ids:
            r_h_string = id2roles_h[r_h_id]
            if 'wikipeople' in args.indir.lower():
                r_t_string = r_h_string.replace("_h", "_t")
            elif 'jf17k' in args.indir.lower():
                r_t_string = r_h_string[:-1] + '1'
            r_t_id = roles_t_2id[r_t_string]
            list_of_roles_t_ids.append(r_t_id)

        with open(args.indir + "/dictionaries_and_facts_permutate.bin", 'rb') as fin:
            data_info1 = pickle.load(fin)
        whole_train = data_info1["train_facts"]
        whole_valid = data_info1["valid_facts"]
        whole_test = data_info1['test_facts']

        model = torch.load(args.outdir)

        epoch = args.outdir.split("/")[-1].split("_")[2]

        print("loading model at epoch", epoch)

        t2.tic()
        output_queue = mp.Queue()
        prepare_data_for_evaluation_and_evaluate_on_multiple_gpus (model, test, epoch, n_roles, n_values, whole_train, whole_test, whole_valid, gpu_ids_splitted, output_queue, args.new_eval_method, list_of_roles_h_ids, list_of_roles_t_ids, n_roles_h, args.dataset_without_h_and_t)
        t2.toc()
        print("Evaluation running time (seconds):", t2.elapsed)

        print("END OF SCRIPT!")

        sys.stdout.flush()

    else:

        # Load training data
        with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
            data_info = pickle.load(fin)
        train = data_info["train_facts"]
        valid = data_info["valid_facts"]
        test = data_info['test_facts']
        values2id = data_info['values_indexes'] #values_indexes
        roles2id = data_info['roles_indexes'] #roles_indexes
        role_val = data_info['role_val']
        value_array = np.array(list(values2id.values()))
        role_array = np.array(list(roles2id.values()))

        id2value = {} #indexes_values
        for tmpkey in values2id:
            id2value[values2id[tmpkey]] = tmpkey
        id2role = {} #indexes_roles
        for tmpkey in roles2id:
            id2role[roles2id[tmpkey]] = tmpkey

        n_values = len(values2id)
        n_roles = len(roles2id)
        print("Unique number of roles:", n_roles)
        print("Unique number of values:", n_values)

        roles_h_2id = {}
        roles_t_2id = {}
        id2roles_h = {}
        id2roles_t = {}
        roleH2roleT = {}
        if 'wikipeople' in args.indir.lower():
            print("\n**************\ncreating extra dictionaries for wikipeople dataset\n**************\n")
            for r in roles2id:
                if r.endswith("_h"):
                    roles_h_2id[r] = roles2id[r]
                    id2roles_h[roles2id[r]] = r
                elif r.endswith("_t"):
                    roles_t_2id[r] = roles2id[r]
                    id2roles_t[roles2id[r]] = r
            for r_h_id in id2roles_h:
                r_h_string = id2roles_h[r_h_id]
                r_t_string = r_h_string.replace("_h", "_t")
                r_t_id = roles_t_2id[r_t_string]
                roleH2roleT[r_h_id] = r_t_id
        elif 'jf17k' in args.indir.lower():
            print("\n**************\ncreating extra dictionaries for JF17K dataset\n**************\n")
            for r in roles2id:
                if r.endswith("0"):
                    roles_h_2id[r] = roles2id[r]
                    id2roles_h[roles2id[r]] = r
                elif r.endswith("1"):
                    roles_t_2id[r] = roles2id[r]
                    id2roles_t[roles2id[r]] = r
            for r_h_id in id2roles_h:
                r_h_string = id2roles_h[r_h_id]
                r_t_string = r_h_string[:-1] + '1'
                r_t_id = roles_t_2id[r_t_string]
                roleH2roleT[r_h_id] = r_t_id


        n_roles_h = len(roles_h_2id)
        n_roles_t = len(roles_t_2id)
        list_of_roles_h_ids = list(id2roles_h.keys())
        list_of_roles_t_ids = []
        for r_h_id in list_of_roles_h_ids:
            r_h_string = id2roles_h[r_h_id]
            if 'wikipeople' in args.indir.lower():
                r_t_string = r_h_string.replace("_h", "_t")
            elif 'jf17k' in args.indir.lower():
                r_t_string = r_h_string[:-1] + '1'
            r_t_id = roles_t_2id[r_t_string]
            list_of_roles_t_ids.append(r_t_id)

        # Load the whole dataset for negative sampling in "batching.py"
        with open(args.indir + "/dictionaries_and_facts_permutate.bin", 'rb') as fin:
            data_info1 = pickle.load(fin)
        whole_train = data_info1["train_facts"]
        whole_valid = data_info1["valid_facts"]
        whole_test = data_info1['test_facts']

        # Prepare validation and test facts
        x_valid = []
        y_valid = []
        for k in valid:
            x_valid.append(np.array(list(k.keys())).astype(np.int32))
            y_valid.append(np.array(list(k.values())).astype(np.float32))
        x_test = []
        y_test = []
        for k in test:
            x_test.append(np.array(list(k.keys())).astype(np.int32))
            y_test.append(np.array(list(k.values())).astype(np.int32))

        #build the model
        if args.model=='HINGE5':
            model = HINGE5(len(roles2id), len(values2id), int(args.embsize), int(args.num_filters), int(args.ngfcn)).cuda()
        else:
            print("--model input parameter wrong (use one of the model implemented in model.py):", args.preprocess)
            sys.exit()

        model.init() #initialize the embeddings with xavier weights initialization

        if args.debug == 'True':
            print("model.emb_roles:", model.emb_roles.weight, "\n\n")
            print("model.emb_values:", model.emb_values.weight, "\n\n")


        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())


        opt = torch.optim.Adam(model.parameters(), lr=float(args.learningrate)) #parameters returns all tensors that represents the parameters of the model
        mp.set_start_method('spawn')
        t1 = TicToc()
        t2 = TicToc()


        n_batches_per_epoch = []
        for i in train:
            ll = len(i)
            if ll == 0:
                n_batches_per_epoch.append(0)
            else:
                n_batches_per_epoch.append(int((ll - 1) / args.batchsize) + 1)

        epoch = 0

        for epoch in range(1, int(args.epochs)+1):
            t1.tic()
            model.train()
            model.to(gpu_ids_splitted[0])
            train_loss = 0
            rel = 0

            for i in range(len(train)): #batch_number == i
                train_i_indexes = np.array(list(train[i].keys())).astype(np.int32)
                train_i_values = np.array(list(train[i].values())).astype(np.float32)

                for batch_num in range(n_batches_per_epoch[i]):

                    arity = i + 2  

                    x_batch, y_batch = Batch_Loader(train_i_indexes, train_i_values, n_values, n_roles, role_val, args.batchsize, arity, whole_train[i], id2value, id2role, args.new_negative_sampling_h_and_t, roleH2roleT, args.new_batching_method, args.num_negative_samples, args.fifty_percent_prob_of_creating_neg_samples, args.dataset_without_h_and_t)

                    pred = model(x_batch, arity, "training", gpu_ids_splitted[0], id2role, id2value)
                    pred = pred * torch.FloatTensor(y_batch).cuda(gpu_ids_splitted[0]) * (-1)
                    loss = model.loss(pred).mean() #Softplus

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss += loss.item()
            t1.toc()
            print("End of epoch", epoch, "- train_loss:", train_loss, "- training time (seconds):", t1.elapsed)

            sys.stdout.flush()

        print("END OF EPOCHS")

        t2.tic()
        output_queue = mp.Queue()
        prepare_data_for_evaluation_and_evaluate_on_multiple_gpus (model, test, epoch, n_roles, n_values, whole_train, whole_test, whole_valid, gpu_ids_splitted, output_queue, id2role, id2value, args.new_eval_method, list_of_roles_h_ids, list_of_roles_t_ids, n_roles_h, args.dataset_without_h_and_t)
        t2.toc()
        print("Evaluation last epoch ", epoch, "- running time (seconds):", t2.elapsed)

        print("END OF SCRIPT!")

        sys.stdout.flush()


if __name__ == '__main__':
    main()
