# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:49:55 2022

@author: AA
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime

from classfication import eval_classification
from TS2vec.dct_func import FFT_for_Period
from ts2vec import TS2Vec
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from forecasting import eval_forecasting
import torch, gc

gc.collect()
torch.cuda.empty_cache()


train_data, train_labels, test_data, test_labels = datautils.load_UEA('ArticularyWordRecognition')

t = time.time()

batch_size = 16
num_samples = batch_size * (train_data.shape[1] // batch_size)


model = TS2Vec(
    input_dims=train_data.shape[-1],
    device='cuda',
    batch_size =16,
    lr = 0.001,
    max_train_length=3000,
    output_dims=320,

)

loss_log = model.fit(
    train_data,
    verbose=True
)
t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

out, eval_res = eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                      eval_protocol='svm')
print(eval_res)