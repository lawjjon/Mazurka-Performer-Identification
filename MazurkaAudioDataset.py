import os
import random
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from torch.utils import data
# from torchaudio_augmentations import ( 
#     RandomResizedCrop,
#     RandomApply,
#     PolarityInversion,
#     Noise,
#     Gain,
#     HighLowPass,
#     Delay,
#     PitchShift,
#     Reverb,
#     Compose,
# )

class MazurkaAudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, target_sample_rate, num_samples, split, num_chunks, is_augmentation):
        self.audio_dir =  audio_dir + '/' if audio_dir else ''
        self.metadata_dir = metadata_dir + '/' if metadata_dir else ''
        self.split = split
        self.annotations_file = self.metadata_dir + '%s_discog_filtered_over_15_performances.csv' % split
        self.annotations = pd.read_csv(self.annotations_file).set_index('opus')
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self.target_sample_rate = target_sample_rate
        # if is_augmentation:
        #     self._get_augmentations()

    def __len__(self):
        return len(self.annotations)

    # def _get_augmentations(self):
    #     transforms = [
    #         RandomResizedCrop(n_samples=self.num_samples),
    #         RandomApply([PolarityInversion()], p=0.8),
    #         RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
    #         RandomApply([Gain()], p=0.2),
    #         RandomApply([HighLowPass(sample_rate=16000)], p=0.8),
    #         RandomApply([Delay(sample_rate=16000)], p=0.5),
    #         RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=16000)], p=0.4),
    #         RandomApply([Reverb(sample_rate=16000)], p=0.3),
    #     ]
    #     self.augmentation = Compose(transforms=transforms)

    def __getitem__(self, index):
        label = self.annotations.iloc[index, 9]

        # read audio, resample and mix to mono
        audio_sample_path = self._get_audio_sample_path(index)
        wav, fs = librosa.load(audio_sample_path, sr=self.target_sample_rate, mono=True) 

        # adjust audio length
        wav = self._adjust_audio_length(wav).astype('float32')

        # data augmentation
        # if self.is_augmentation:
        #     wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        return wav, label

    def _adjust_audio_length(self, wav):
      if self.split == 'train':
        random_index = random.randint(0, len(wav) - self.num_samples - 1) 
        wav = wav[random_index : random_index + self.num_samples]
      else:
        hop = (len(wav) - self.num_samples) // self.num_chunks # when num_chunks=1, hop is not actually used
        wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)]) 
      return wav

    def _get_audio_sample_path(self, index):
      piece_path = self.annotations.iloc[index, 8]
      sample_path = self.audio_dir + piece_path + '.mp3'
      return sample_path
