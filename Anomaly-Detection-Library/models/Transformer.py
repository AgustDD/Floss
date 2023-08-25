import torch
import torch.nn as nn
import torch.nn.functional as F

from dct_func import FFT_for_Period1
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

from losses import hierarchical_contrastive_loss, context_sampling


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)



    def anomaly_detection(self, x_enc):
        # Embedding
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
        enc_out1, attns = self.encoder(enc_out1, attn_mask=None)
        enc_out2 = self.enc_embedding(input2, None)  # [B,T,C]
        enc_out2, attns = self.encoder(enc_out2, attn_mask=None)
        out1 = enc_out1[:, -(crop_l + (period_move1 * periodicity)):]
        out1 = out1[:, -(crop_l):]

        out2 = enc_out2[:, (period_move2 * periodicity):crop_l + (period_move2 * periodicity)]

        floss = hierarchical_contrastive_loss(
            out1,
            out2
        )
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return floss, dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        floss, dec_out = self.anomaly_detection(x_enc)
        return floss, dec_out  # [B, L, D]

