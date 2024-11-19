# Cost-Sensitive Learning for Long-Tail Temporal Action Segmentation 

This repository is the official implementation of our paper "Cost-Sensitive Learning for Long-Tailed Temporal Action Segmentation" with model MSTCN on Breakfast dataset.

Please check our [project page](https://pangzhan27.github.io/csl-tas.github.io/) for more information.
## Dataset
The dataset used in our paper is open-source data. It can be downloaded from the references in the main paper. 

## Training
To train the models in the paper, run this command:

```train
python transit-csl.py --dataset breakfast --action train --split 1 --seed 42 --tau 0.5
```

You can find the scripts for other long-tailed methods in the code folder.

## Evaluation

For evaluation, you need first generate the prediction for test set. For example,
```eval
python transit-csl.py --dataset breakfast --action predict --split 1 --seed 42 --tau 0.5 --suf _ncm
```
Then, use the 'eval.py' script for evaluation. The results contain both global and per class metrics.

```eval
python eval.py --method prediction_path
```




 
