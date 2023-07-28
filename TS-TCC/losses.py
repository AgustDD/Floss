import torch
import numpy as np
import torch.nn.functional as F

from dct_func import dct, p_fft


def hierarchical_contrastive_loss(z1, z2, alpha=0, k = 2, f_weight=0.5, temporal_unit=0, beta=0.5, trans_type='dct'):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if trans_type == 'dct':
                loss += beta * freqency_loss(z1, z2)
            elif trans_type == 'fft':
                loss += beta * periogram_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=k).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=k).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d * f_weight


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def freqency_loss(z1, z2):
    o1 = z1.permute( [0, 2, 1])
    o2 = z2.permute([0, 2, 1])
    return torch.mean(torch.abs(torch.abs(dct(o1)) - torch.abs(dct(o2))))


def periogram_loss(z1, z2):
    o1 = z1.permute([0, 2, 1])
    o2 = z2.permute( [0, 2, 1])
    return torch.mean(torch.abs((p_fft(o1)) - (p_fft(o2))))

def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]

def context_sampling(x, temporal_unit):  #BxTxC
    ts_l = x.size(1)
    crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
    crop_left = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l
    crop_eleft = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
    input1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
    input2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
    return input1, input2, crop_l

