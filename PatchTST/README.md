# PatchTST (ICLR 2023)

### This is an offical implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730).

:triangular_flag_on_post: Our model has been included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributor @[kdgutier](https://github.com/kdgutier) and @[cchallu](https://github.com/cchallu)!

:triangular_flag_on_post: Our model has been included in [timeseriesAI(tsai)](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb). Special thanks to the contributor @[oguiza](https://github.com/oguiza)!

:triangular_flag_on_post: We offer a video that provides a concise overview of our paper for individuals seeking a rapid comprehension of its contents: https://www.youtube.com/watch?v=Z3-NrohddJw



## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Requirements
- numpy
- matplotlib
- pandas
- scikit-learn
- torch==1.11.0

## Datasets

You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./data/dataset``` and put all the csv files in the directory.


```

## Getting Started

### Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt patchtst_pretrain.py is to train the PatchTST/64. To run the code with a single GPU on ettm1, just run the following command
   dest can be set to: `ettm1`, `ettm2`, `etth1`, `etth2`, `electricity`, `illness`, `weather`, `exchange`
```
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the patchtst_pretrain.py script.
 
 3. Fine-tuning: The script patchtst_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

## Contact

If you have any questions or concerns, please contact us: ynie@princeton.edu or nnguyen@us.ibm.com or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and 
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

