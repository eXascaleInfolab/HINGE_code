# HINGE (Hyper-relational Knowledge Graph Embedding)

HINGE is a hyper-relational KG embedding model, which directly learns from hyper-relational facts in a KG. HINGE captures not only the primary structural information of the KG encoded in the triplets, but also the correlation between each triplet and its associated key-value pairs. Please see the details in our paper below:
- Paolo Rosso, Dingqi Yang and Philippe Cudre-Mauroux, Beyond Triplets: Hyper-Relational Knowledge Graph Embedding for Link Prediction, In Proc. of The Web  Conference (WWW'20). April 2020, Taipei.
​
## How to run the code
###### Data preprocessing
```
python builddata.py --data_dir <PATH>/<DATASET>/
```
###### Train and evaluate model (suggested parameters for both JF17k and Wiki dataset)
```
python main_hinge.py --indir=<PATH>/<DATASET>/ --epochs=1000 --batchsize=128 --embsize=100 --learningrate=0.0001 --outdir=<PATH>/<DATASET>/ --load=False --num_negative_samples 1  --gpu_ids=0,1,2,3 --num_filters=400
```

###### Parameter setting:
In `main_hinge.py`, you can set:
`--indir`: input file directory
`--epochs`: number of training epochs (suggested: JF17k 600 epochs, Wiki X epochs)
`--batchsize`: batch size of training set
`--embsize`: embedding size
`--learningrate`: learning rate
`--outdir`: where to store HINGE model
`--load`: load a pre-trained HINGE model and evaluate
`--num_negative_samples`: number of negative samples
`--gpu_ids`: gpu to be used for train and test the model
`--num_filters`: number of filters used in the CNN
​
# Python lib versions
Python: 3.6.10
torch: 1.4.0
numpy: 1.18.1
tensorflow-gpu: 2.2.0
​
# Reference
If you use our code or datasets, please cite:
```
@inproceedings{rosso2020beyond,
  title={Beyond triplets: hyper-relational knowledge graph embedding for link prediction},
  author={Rosso, Paolo and Yang, Dingqi and Cudr{\'e}-Mauroux, Philippe},
  booktitle={Proceedings of The Web Conference 2020},
  pages={1885--1896},
  year={2020}
}
```
​
​
# NaLP and NaLP-fix (Baselines used in our paper)
We implemented a fast version of NaLP (Guan, Saiping, et al. "Link prediction on n-ary relational data." WWW 2019). Our implementation is significant faster (about 50x speedup) than its orginal release by maximially exploiting matrix operations on GPUs. We also developed an improved version of it, called NaLP-fix, with a different negative sampling technique from the one used in the original paper. For more details please check our HINGE paper, where we use both NaLP and NaLP-fix as baselines.

## How to run the code
###### Data preprocessing
```
python builddata.py --data_dir <PATH>/<DATASET>/
python builddata.py --data_dir <PATH>/<DATASET>/ --if_permutate True --bin_postfix _permutate
```
###### Train and evaluate NaLP (suggested parameters for JF17k dataset)
```
python main_nary_pytorch.py --indir=<PATH>/JF17K_original/ --epochs=2000 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=<PATH>/JF17K_original/ --load=False --gpu_ids=0,1,2,3 --ngfcn 1000 --new_negative_sampling_h_and_t False --num_negative_samples 1
```
###### Train and evaluate NaLP-fix (suggested parameters for JF17k dataset)
```
python main_nary_pytorch.py --indir=<PATH>/JF17K_original/ --epochs=2000 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=<PATH>/JF17K_original/ --load=False --gpu_ids=0,1,2,3 --ngfcn 1000 --new_negative_sampling_h_and_t True --num_negative_samples 1
```
###### Train and evaluate NaLP (suggested parameters for Wiki dataset)
```
python main_nary_pytorch.py --indir=<PATH>/wikipeople_original/ --epochs=600 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=<PATH>/wikipeople_original/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1200 --new_negative_sampling_h_and_t False --num_negative_samples 1
```
###### Train and evaluate NaLP-fix (suggested parameters for Wiki dataset)
```
python main_nary_pytorch.py --indir=<PATH>/wikipeople_original/ --epochs=500 --batchsize=128 --embsize=100 --learningrate=0.00005 --outdir=<PATH>/wikipeople_original/ --load=False --gpu_ids=5,6,7,8 --ngfcn 1200 --new_negative_sampling_h_and_t True --num_negative_samples 1
```


# Reference
If you use our code or datasets, please cite:
```
@inproceedings{rosso2020beyond,
  title={Beyond triplets: hyper-relational knowledge graph embedding for link prediction},
  author={Rosso, Paolo and Yang, Dingqi and Cudr{\'e}-Mauroux, Philippe},
  booktitle={Proceedings of The Web Conference 2020},
  pages={1885--1896},
  year={2020}
}
```
