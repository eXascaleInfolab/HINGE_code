#!/bin/bash
python builddata.py --data_dir /mnt/hdd/paolo/data/property_graph/JF17K_version1_test_25_percent_ORIGINAL/
python builddata.py --data_dir /mnt/hdd/paolo/data/property_graph/JF17K_version1_test_25_percent_ORIGINAL/ --if_permutate True --bin_postfix _permutate
python main_nary_pytorch.py --indir=/mnt/hdd/paolo/data/property_graph/JF17K_version1_test_25_percent_ORIGINAL/ --model=HINGE5 --epochs=1000 --batchsize=128 --embsize=100 --learningrate=0.0001 --debug=False --outdir=/mnt/hdd/paolo/data/property_graph/JF17K_version1_test_25_percent_ORIGINAL/ --load=False --ngfcn 200 --new_eval_method True --new_negative_sampling_h_and_t True --new_batching_method True --num_negative_samples 1 --dataset_without_h_and_t True  --gpu_ids=0,1,2,3 --num_filters=400 --fifty_percent_prob_of_creating_neg_samples True > log_JFK_hinge5_neg1_nun_fil400_bat128_lr0001_er50.txt


python builddata.py --data_dir /mnt/hdd/paolo/data/property_graph/wikipeople_test_25_percent_ORIGINAL/
python builddata.py --data_dir /mnt/hdd/paolo/data/property_graph/wikipeople_test_25_percent_ORIGINAL/ --if_permutate True --bin_postfix _permutate
python main_nary_pytorch.py --indir=/mnt/hdd/paolo/data/property_graph/wikipeople_test_25_percent_ORIGINAL/ --model=HINGE5 --epochs=1000 --batchsize=128 --embsize=100 --learningrate=0.0001 --debug=False --outdir=/mnt/hdd/paolo/data/property_graph/wikipeople_test_25_percent_ORIGINAL/ --load=False --ngfcn 200 --new_eval_method True --new_negative_sampling_h_and_t True --new_batching_method True --num_negative_samples 1 --dataset_without_h_and_t True  --gpu_ids=0,1,2,3  --num_filters=400 --fifty_percent_prob_of_creating_neg_samples True > log_WIKI_hinge5_neg1_nun_fil400_bat128_lr00005_er50.txt


echo "\n\nEND BASH\n\n"
