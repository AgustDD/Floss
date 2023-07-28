import math
import os
import sys

from ..dct_func import FFT_for_Period

sys.path.append("..")
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.loss import NTXentLoss
from ..losses import hierarchical_contrastive_loss, context_sampling


def Trainer(model, model2, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl,
            device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model2, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, model2, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader,
                config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    def floss(data):
        data = torch.transpose(data, 1, 2)  # BxCxT -> BxTxC
        periodicity, freq_list = FFT_for_Period(data, 1)
        periodicity = torch.from_numpy(np.array([periodicity]))
        periodicity = periodicity.item()

        input1, input2, crop_l = context_sampling(data, 0)

        if input1.shape[1] - crop_l > periodicity and input2.shape[
            1] - crop_l > periodicity and periodicity > 0:
            period_move1 = random.randint(0, (input1.shape[1] - crop_l) // periodicity)
            period_move2 = random.randint(0, (input2.shape[1] - crop_l) // periodicity)
        else:
            period_move1 = 0
            period_move2 = 0

        input1 = torch.transpose(input1, 1, 2)  # BxCxT
        input2 = torch.transpose(input2, 1, 2)

        out1 = model2(input1)  # BxCxT
        out1 = torch.transpose(out1, 1, 2)  # BxTxC
        out1 = out1[:, -(crop_l + (period_move1 * periodicity)):]
        out1 = out1[:, -(crop_l):]

        out2 = model2(input2)
        out2 = torch.transpose(out2, 1, 2)
        out2 = out2[:, (period_move2 * periodicity):crop_l + (period_move2 * periodicity)]
        length_diff = out1.size(1) - out2.size(1)  # 计算长度差异

        if length_diff > 0:
            # 如果 out1 的长度大于 out2，对 out1 进行切割
            out1 = out1[:, :out2.size(1)]
        elif length_diff < 0:
            # 如果 out2 的长度大于 out1，对 out2 进行切割
            out2 = out2[:, :out1.size(1)]

        # 确保 out1 和 out2 的长度相同
        assert out1.size(1) == out2.size(1)
        floss = hierarchical_contrastive_loss(
            out1,
            out2
        )
        # print("floss:", floss)
        if not math.isnan(floss):
            return floss
        else:
            return 0

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)  # BxCxT
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2

            floss1 = floss(aug1)
            floss2 = floss(aug2)
            # print(floss1+floss2)

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2 + (
                        floss1 + floss2)

        else:  # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
