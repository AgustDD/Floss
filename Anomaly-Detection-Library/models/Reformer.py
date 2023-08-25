import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dct_func import FFT_for_Period1
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding
from losses import hierarchical_contrastive_loss, context_sampling


class Model(nn.Module):
    """
    Reformer with O(LlogL) complexity
    Paper link: https://openreview.net/forum?id=rkgNKkHtvB
    """

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        """
        bucket_size: int, 
        n_hashes: int, 
        """
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads,
                                  bucket_size=bucket_size, n_hashes=n_hashes),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )



        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)


    def anomaly_detection(self, x_enc):

        periodicity, freq_list = FFT_for_Period1(x_enc, 1)  # 初始周期性
        periodicity = torch.from_numpy(np.array([periodicity]))
        periodicity = periodicity.item()
        input1, input2, crop_l = context_sampling(x_enc, 0)
        if input1.shape[1] - crop_l > periodicity and input2.shape[
            1] - crop_l > periodicity and periodicity > 0:
            period_move1 = np.random.randint(0, (input1.shape[1] - crop_l) // periodicity)
            period_move2 = np.random.randint(0, (input2.shape[1] - crop_l) // periodicity)
        else:
            period_move1 = 0
            period_move2 = 0
            # embedding

        enc_out1 = self.enc_embedding(input1, None)  # [B,T,C]
        enc_out1, attns = self.encoder(enc_out1)
        enc_out2 = self.enc_embedding(input2, None)  # [B,T,C]
        enc_out2, attns = self.encoder(enc_out2)
        out1 = enc_out1[:, -(crop_l + (period_move1 * periodicity)):]
        out1 = out1[:, -(crop_l):]

        out2 = enc_out2[:, (period_move2 * periodicity):crop_l + (period_move2 * periodicity)]

        floss = hierarchical_contrastive_loss(
            out1,
            out2
        )

        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        enc_out, attns = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return floss, enc_out  # [B, L, D]



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        floss, dec_out = self.anomaly_detection(x_enc)
        return floss, dec_out  # [B, L, D]

