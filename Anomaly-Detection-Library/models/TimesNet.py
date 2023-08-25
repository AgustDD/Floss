import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
import numpy as np
from dct_func import FFT_for_Period1
from losses import context_sampling, hierarchical_contrastive_loss


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)


    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()

        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        periodicity, freq_list = FFT_for_Period1(x_enc, 1)
        periodicity = torch.from_numpy(np.array([periodicity]))
        periodicity = periodicity.item()

        input1, input2, crop_l = context_sampling(x_enc, 0)
        #print(crop_l)
        if input1.shape[1] - crop_l > periodicity and input2.shape[
            1] - crop_l > periodicity and periodicity > 0:
            period_move1 = np.random.randint(0, (input1.shape[1] - crop_l) // periodicity)
            period_move2 = np.random.randint(0, (input2.shape[1] - crop_l) // periodicity)
        else:
            period_move1 = 0
            period_move2 = 0
        # embedding
        enc_out1 = self.enc_embedding(input1, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out1 = self.layer_norm(self.model[i](enc_out))

        enc_out2 = self.enc_embedding(input2, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out2 = self.layer_norm(self.model[i](enc_out))

        out1 = enc_out1[:, -(crop_l + (period_move1 * periodicity)):]
        out1 = out1[:, -(crop_l):]

        out2 = enc_out2[:, (period_move2 * periodicity):crop_l + (period_move2 * periodicity)]
        #print("out1:", out1.size())
        #print("out2:", out2.size())
        length_diff = out1.size(1) - out2.size(1)

        if length_diff > 0:
            out1 = out1[:, :out2.size(1)]
        elif length_diff < 0:
            out2 = out2[:, :out1.size(1)]

        assert out1.size(1) == out2.size(1)

        floss = hierarchical_contrastive_loss(
            out1,
            out2
        )
        #print("floss:", floss)

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return floss, dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        floss, dec_out = self.anomaly_detection(x_enc)
        return floss, dec_out  # [B, L, D]
