#!/bin/bash
python builddata.py --data_dir data/JF17K/
python builddata.py --data_dir data/JF17K/ --if_permutate True --bin_postfix _permutate
taskset --cpu-list 10-19 python main_nary_pytorch.py --indir=data/JF17K/ --epochs=500 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=data/JF17K/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1000 --new_negative_sampling_h_and_t True --num_negative_samples 1 > nohup_NaLP_FIX_JF17K.out
taskset --cpu-list 10-19 python main_nary_pytorch.py --indir=data/JF17K/ --epochs=600 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=data/JF17K/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1000 --new_negative_sampling_h_and_t False --num_negative_samples 1 > nohup_NaLP_JF17K.out

python builddata.py --data_dir data/wikipeople/
python builddata.py --data_dir data/wikipeople/ --if_permutate True --bin_postfix _permutate
taskset --cpu-list 10-19 python main_nary_pytorch.py --indir=data/wikipeople/ --epochs=500 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=data/wikipeople/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1200 --new_negative_sampling_h_and_t True --num_negative_samples 1 > nohup_NaLP_FIX_wikipeople.out
taskset --cpu-list 10-19 python main_nary_pytorch.py --indir=data/wikipeople/ --epochs=600 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=data/wikipeople/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1200 --new_negative_sampling_h_and_t False --num_negative_samples 1 > nohup_NaLP_wikipeople.out

echo "\n\nEND BASH\n\n"
