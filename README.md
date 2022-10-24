# BASQ: Branch-wise Activation-clipping Search Quantization for Sub-4-bit Neural Networks

### [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720017.pdf) | [Supplementary](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720017-supp.pdf)

BASQ: Branch-wise Activation-clipping Search Quantization for Sub-4-bit Neural Networks, ECCV 2022 <br>
 [Han-Byul Kim](https://www.linkedin.com/in/han-byul-kim-336479253/)<sup>1</sup>,
 [Eunhyeok Park]()<sup>2</sup>,
 [Sungjoo Yoo]()<sup>1</sup> <br>
 <sup>1</sup>Seoul National University, <sup>2</sup>POSTECH

## What is BASQ?
Accurate 2-bit to 4-bit uniform quantization method for both MobileNets and ResNets working with

① Quantization hyper-parameter (clipping value with L2-decay weight) search with architecture search method

② Block structure for removing unstable effect from proposed quantization hyper-parameter search

The search process of BASQ is composed of three steps as below.

## 01. Supernet training
```console
$ python 01train_supernet.py --data {your_imagenet_dataset_folder}
```
## 02. Architecture search
```console
$ python 02kfold_evolution.py --data {your_imagenet_dataset_folder}
```
## 03. Finetuning
Before running finetuning, please attach the result of architecture search as below.

For example, our results of architecture search are same as below.

```
===total-result===
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fold_accs: [67.62, 66.2, 64.86, 68.94, 55.18, 58.96, 54.82, 56.46, 53.54, 63.48]
full_accs: [59.30222222222222, 59.29333333333334, 59.26222222222222, 58.67777777777778, 59.733333333333334, 59.15777777777778, 60.01111111111111, 60.45333333333333, 60.10888888888889, 59.53333333333333]
archs: [(5, 3, 2, 2, 1, 1, 6, 6, 1, 2, 4, 6, 0, 3, 3, 5, 5), (1, 4, 5, 0, 5, 2, 3, 6, 4, 5, 6, 6, 6, 6, 3, 2, 5), (1, 4, 5, 0, 2, 5, 5, 2, 5, 2, 1, 1, 2, 2, 6, 4, 5), (1, 4, 5, 1, 1, 2, 4, 6, 2, 1, 2, 6, 4, 2, 6, 4, 5), (1, 4, 5, 5, 1, 5, 1, 5, 6, 5, 4, 1, 6, 4, 3, 2, 3), (5, 3, 2, 2, 6, 6, 5, 0, 5, 5, 6, 6, 4, 6, 5, 0, 2), (1, 4, 5, 1, 6, 2, 6, 4, 2, 1, 1, 6, 4, 0, 6, 4, 5), (5, 3, 2, 2, 1, 1, 4, 2, 1, 2, 4, 6, 0, 0, 3, 6, 5), (1, 4, 5, 5, 6, 5, 2, 3, 5, 6, 3, 6, 5, 2, 6, 0, 2), (5, 3, 2, 2, 4, 1, 4, 2, 1, 4, 6, 6, 0, 4, 3, 2, 5)]
```

Paste it to line 66 of 03finetuning.py.

```
FOLD_ACCS = [67.62, 66.2, 64.86, 68.94, 55.18, 58.96, 54.82, 56.46, 53.54, 63.48]
FULL_ACCS = [59.30222222222222, 59.29333333333334, 59.26222222222222, 58.67777777777778, 59.733333333333334, 59.15777777777778, 60.01111111111111, 60.45333333333333, 60.10888888888889, 59.53333333333333]
BEST_ARCHS = [(5, 3, 2, 2, 1, 1, 6, 6, 1, 2, 4, 6, 0, 3, 3, 5, 5), (1, 4, 5, 0, 5, 2, 3, 6, 4, 5, 6, 6, 6, 6, 3, 2, 5), (1, 4, 5, 0, 2, 5, 5, 2, 5, 2, 1, 1, 2, 2, 6, 4, 5), (1, 4, 5, 1, 1, 2, 4, 6, 2, 1, 2, 6, 4, 2, 6, 4, 5), (1, 4, 5, 5, 1, 5, 1, 5, 6, 5, 4, 1, 6, 4, 3, 2, 3), (5, 3, 2, 2, 6, 6, 5, 0, 5, 5, 6, 6, 4, 6, 5, 0, 2), (1, 4, 5, 1, 6, 2, 6, 4, 2, 1, 1, 6, 4, 0, 6, 4, 5), (5, 3, 2, 2, 1, 1, 4, 2, 1, 2, 4, 6, 0, 0, 3, 6, 5), (1, 4, 5, 5, 6, 5, 2, 3, 5, 6, 3, 6, 5, 2, 6, 0, 2), (5, 3, 2, 2, 4, 1, 4, 2, 1, 4, 6, 6, 0, 4, 3, 2, 5)]
```

Run finetuning.

```console
$ python 03finetune.py --data {your_imagenet_dataset_folder}
```

Final results are printed as below. Bnfull_accs are finetuned results.
```
===final-result===
fold_accs: [67.62, 66.2, 64.86, 68.94, 55.18, 58.96, 54.82, 56.46, 53.54, 63.48]
full_accs: [59.30222222222222, 59.29333333333334, 59.26222222222222, 58.67777777777778, 59.733333333333334, 59.15777777777778, 60.01111111111111, 60.45333333333333, 60.10888888888889, 59.53333333333333]
**bnfull_accs: [63.95, 64.02, 64.28, 63.89, 65.46, 65.1, 65.23, 65.32, 65.4, 64.49]
```

## Checkpoint

Supernet training checkpoint is in train-supernet folder with named as 'model_actwbin_best_466000it.pth'.

## Citation

```
@inproceedings{basq,
  title={BASQ: Branch-wise Activation-clipping Search Quantization for Sub-4-bit Neural Networks},
  author={Han-Byul Kim and Eunhyeok Park and Sungjoo Yoo},
  year={2022},
  booktitle={ECCV},
}
```