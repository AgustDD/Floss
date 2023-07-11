# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:49:55 2022

@author: AA
"""

from ts2vec import TS2Vec
import datautils
from forecasting import eval_forecasting
import torch, gc

gc.collect()
torch.cuda.empty_cache()

data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv('national_illness',
                                                                                                              univar=False)
train_data = data[:, train_slice]

model = TS2Vec(
    input_dims=train_data.shape[-1],
    device='cuda',
    batch_size = 2,
    lr = 0.001,
    max_train_length=3000,
    output_dims=320,
    #periodicity= periodicity
)

loss_log = model.fit(
    train_data,
    verbose=True
)

out, eval_res = eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
print(eval_res)



