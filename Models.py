# contains code from https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py
# all sigmoid activation functions are commented out, because they are not used in the loss function

import numpy as np
import torch
from torch import nn
import torchaudio

from modules import Conv_1d, Conv_2d, HarmonicSTFT, Res_2d_mp, Res_2d, Conv_V, Conv_H


## takes 29s input length mel-spectrogram (29*16000)
class FCN(nn.Module): 
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0, 
                f_max=8000.0, 
                n_mels=96,
                n_class=50):
        super(FCN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2,4))
        self.layer2 = Conv_2d(64, 128, pooling=(2,4))
        self.layer3 = Conv_2d(128, 128, pooling=(2,4))
        self.layer4 = Conv_2d(128, 128, pooling=(3,5))
        self.layer5 = Conv_2d(128, 64, pooling=(4,4))

        # Dense
        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        #x = nn.Sigmoid()(x)

        return x
    

# 3s input length mel-spectrogram (3*16000)
class Musicnn(nn.Module):
    '''
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0, 
                f_max=8000.0, 
                n_mels=96,
                n_class=50,
                dataset='mtat'):
        super(Musicnn, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons front-end
        m1 = Conv_V(1, 204, (int(0.7*96), 7))
        m2 = Conv_V(1, 204, (int(0.4*96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel= 512 if dataset=='msd' else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        dense_channel = 500 if dataset=='msd' else 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, n_class)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        #out = nn.Sigmoid()(out)

        return out


# takes 5s input length (5*16000) stacked harmonic tensor
class HarmonicCNN(nn.Module): 
    '''
    Won et al. 2020
    Data-driven harmonic filters for audio representation learning.
    Trainable harmonic band-pass filters, short-chunk CNN.
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0, 
                f_max=8000.0, 
                n_mels=128,
                n_class=50,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw='only_Q'):
        super(HarmonicCNN, self).__init__()

        # Harmonic STFT
        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_harmonic=n_harmonic,
                                  semitone_scale=semitone_scale,
                                  learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # CNN
        self.layer1 = Conv_2d(n_harmonic, n_channels, pooling=2)
        self.layer2 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer3 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer4 = Res_2d_mp(n_channels, n_channels, pooling=2)
        self.layer5 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer6 = Res_2d_mp(n_channels*2, n_channels*2, pooling=(2,3))
        self.layer7 = Res_2d_mp(n_channels*2, n_channels*2, pooling=(2,3))

        # Dense
        self.dense1 = nn.Linear(n_channels*2, n_channels*2)
        self.bn = nn.BatchNorm1d(n_channels*2)
        self.dense2 = nn.Linear(n_channels*2, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.hstft_bn(self.hstft(x))

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        #x = nn.Sigmoid()(x)

        return x
    
  
# takes 3.69s input length mel-spectrogram (num_samples=59049)
class ShortChunkCNNRes(nn.Module):
  def __init__(self, n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0, 
                f_max=8000.0, 
                n_mels=128,
                n_class=50): 
    super().__init__()

    # Spectrogram
    self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                      n_fft=n_fft,
                                                      f_min=f_min,
                                                      f_max=f_max,
                                                      n_mels=n_mels)
    self.to_db = torchaudio.transforms.AmplitudeToDB()
    self.spec_bn = nn.BatchNorm2d(1)

    # CNN
    self.layer1 = Res_2d(1, n_channels, stride=2)
    self.layer2 = Res_2d(n_channels, n_channels, stride=2)
    self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
    self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
    self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
    self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
    self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

    # Dense
    self.dense1 = nn.Linear(n_channels*4, n_channels*4)
    self.bn = nn.BatchNorm1d(n_channels*4)
    self.dense2 = nn.Linear(n_channels*4, n_class)
    self.dropout = nn.Dropout(0.5)
    self.relu = nn.ReLU()
  
  def forward(self, x):
    # Spectrogram
    x = self.spec(x)
    x = self.to_db(x)
    x = x.unsqueeze(1)
    x = self.spec_bn(x)

    # CNN
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = x.squeeze(2)

    # Global Max Pooling
    if x.size(-1) != 1:
        x = nn.MaxPool1d(x.size(-1))(x)
    x = x.squeeze(2)

    # Dense
    x = self.dense1(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.dense2(x)
    #x = nn.Sigmoid()(x)
    
    return x
  

  
### CNNSA (CNN with self-attention), called "self-attention" in the readme, "attention" in trained models ###
## takes 15s input length mel-spectrogram (15*16000)
## self-attention usually helps
## only difference to CRNN is that self-attention mechanism is used instead of the RNNs
## uses transformer encoder

from attention_modules import BertConfig, BertEncoder, BertPooler

class CNNSA(nn.Module):
    '''
    Won et al. 2019
    Toward interpretable music tagging with self-attention.
    Feature extraction with CNN + temporal summary with Transformer encoder.
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0, 
                f_max=8000.0, 
                n_mels=128,
                n_class=50): 
        super(CNNSA, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))
        self.layer7 = Res_2d(n_channels*2, n_channels*2, stride=(2, 1))

        # Transformer encoder
        bert_config = BertConfig(vocab_size=256,
                                 hidden_size=256,
                                 num_hidden_layers=2,
                                 num_attention_heads=8,
                                 intermediate_size=1024,
                                 hidden_act="gelu",
                                 hidden_dropout_prob=0.4,
                                 max_position_embeddings=700,
                                 attention_probs_dropout_prob=0.5)
        self.encoder = BertEncoder(bert_config)
        self.pooler = BertPooler(bert_config)
        self.vec_cls = self.get_cls(256)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(256, n_class)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Get [CLS] token
        x = x.permute(0, 2, 1)
        x = self.append_cls(x)

        # Transformer encoder
        x = self.encoder(x)
        x = x[-1]
        x = self.pooler(x)

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        #x = nn.Sigmoid()(x)

        return x