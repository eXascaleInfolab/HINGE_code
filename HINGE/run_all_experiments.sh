#!/bin/bash

python builddata.py --data_dir data/JF17K/
python builddata.py --data_dir v --if_permutate True --bin_postfix _permutate
taskset --cpu-list 0-9 python main_hinge.py --indir=data/JF17K/ --epochs=600 --batchsize=128 --embsize=100 --learningrate=0.0001 --outdir=data/JF17K/ --load=False --num_negative_samples 1  --gpu_ids=1,2,3,4 --num_filters=400 > log_JFK_hinge.txt

python builddata.py --data_dir data/wikipeople/
python builddata.py --data_dir data/wikipeople/ --if_permutate True --bin_postfix _permutate
taskset --cpu-list 10-19 python main_hinge.py --indir=data/wikipeople/ --epochs=400 --batchsize=128 --embsize=100 --learningrate=0.0001 --outdir=data/wikipeople/ --load=False --num_negative_samples 1 --gpu_ids=5,6,7,8 --num_filters=400 > log_WIKI_hinge.txt

echo "\n\nEND BASH\n\n"
