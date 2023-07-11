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

from anomaly_detection import eval_anomaly_detection, eval_anomaly_detection_coldstart
from ts2vec import TS2Vec
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from forecasting import eval_forecasting
import torch, gc

gc.collect()
torch.cuda.empty_cache()

all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(
    'yahoo')
#train_data = datautils.gen_ano_train_data(all_train_data)
#train_data, _, _, _ = datautils.load_UCR('ItalyPowerDemand')
data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv('electricity',
                                                                                                              univar=False)
train_data = data[:, train_slice]
t = time.time()
model = TS2Vec(
    input_dims=train_data.shape[-1],
    device='cuda',
    batch_size = 16 ,
    lr = 0.001,
    max_train_length=3000,
    output_dims=320
)

loss_log = model.fit(
    train_data,
    verbose=True
)
t = time.time() - t
print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
out, eval_res = eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps,
                                                         all_test_data, all_test_labels, all_test_timestamps, delay)
print(eval_res)