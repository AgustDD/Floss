import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.control.control_plots import np

from dct_func import FFT_for_Period1
from layers.Embed import DataEmbedding
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from losses import context_sampling, hierarchical_contrastive_loss


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )


        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)



    def anomaly_detection(self, x_enc):
        # enc
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
        enc_out1, attns1 = self.encoder(enc_out1, attn_mask=None)
        enc_out2 = self.enc_embedding(input2, None)  # [B,T,C]
        enc_out2, attns2 = self.encoder(enc_out2, attn_mask=None)
        out1 = enc_out1[:, -(crop_l + (period_move1 * periodicity)):]
        out1 = out1[:, -(crop_l):]

        out2 = enc_out2[:, (period_move2 * periodicity):crop_l + (period_move2 * periodicity)]

        floss = hierarchical_contrastive_loss(
            out1,
            out2
        )

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return floss, dec_out



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        floss, dec_out = self.anomaly_detection(x_enc)
        return floss, dec_out  # [B, L, D]

