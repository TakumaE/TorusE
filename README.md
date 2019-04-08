# TorusE
An embedding model onto a torus for knowledge graph completion.

Paper: [TorusE: Knowledge Graph Embedding on a Lie Group](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16227)

```
@inproceedings{TorusE,
  author    = {Takuma Ebisu and
               Ryutaro Ichise},
  title     = {TorusE: Knowledge Graph Embedding on a Lie Group},
  booktitle = {Proceedings of the Thirtieth {AAAI} Conference on Artificial Intelligence},
  year      = {2018},
}
```


## Accuracy
Dataset | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---:
WN18 | 0.947 | 0.943 | 0.950 | 0.954
FB15k | 0.747 | 0.690 | 0.785 | 0.840

The results on FB15k is slightly better than the results in the paper. This is because there was a bug with the eL2 distance function in the original implementation. According to fix, we retuned hyperparameters for FB15k.


## Requirement
Tensorflow
Numpy

## Data Format
Datasets for this implementation should have three files named as following: train, valid, and test.
You need to put under the directory, data/datasets_name/.
Each line in these files represent a triple. For example, a line in a file, son sibling_of daughter, represents the triple (son, sibling_of, daughter).

Example data are in data/example/.

## Reproduction of the results
1. Put the datasets WN18 and FB15k under ./data/.

2a. run the following command for FB15k
```
python run.py -reproduce transe-fb15k
```

2b. run the following command for WN18
```
python run.py -reproduce transe-wn18
```

## Acknowledgement
I really appreciate [Phuc Nguyen](https://github.com/phucty). He helped me to reconstruct my code for readability.
