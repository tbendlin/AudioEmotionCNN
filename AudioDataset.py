"""
    File containing custom implementation for a PyTorch Dataset

    :author Theodora Bendlin
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import isfile
from os import listdir
from Preprocessor import *


class AudioDataset(Dataset):
    def __init__(self, raw_data_root, saved_data_root, noise_reps, transforms=None):
        self.samples = []
        self.labels = []
        self.raw_data_root = raw_data_root
        self.saved_data_root = saved_data_root
        self.noise_reps = noise_reps
        self.transforms = transforms

        for voice_sample in listdir(raw_data_root):
            if voice_sample.startswith("."):
                continue

            label = self.__getclasslabel__(voice_sample)

            self.labels.append(label)
            self.samples.append(voice_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.__getspectogramdata__(self.samples[idx])
        if self.transforms:
            sample = self.transforms(sample)

        return sample, self.labels[idx]

    def __getspectogramdata__(self, file_name):

        file_name_stripped = file_name.split(".wav")[0]
        extension = "--size-129-noise-" + str(self.noise_reps) + ".npy"

        if isfile(self.saved_data_root + file_name_stripped + extension):
            return np.load(self.saved_data_root + file_name_stripped + extension)

        spectrogram = preprocess_and_create_spectrogram(self.raw_data_root + file_name, noise_reps=self.noise_reps)

        # Save the spectogram for easier loading
        spectrogram = spectrogram.astype(np.float32)
        np.save(self.saved_data_root + file_name_stripped + extension, spectrogram)

        return spectrogram

    def __getclasslabel__(self, file_name):
        name_parts = file_name.split("--")
        emotion = name_parts[1]

        if emotion.startswith("n") or emotion.startswith("03-01-01") or "neutral" in emotion:
            return 0
        elif emotion.startswith("h") or emotion.startswith("03-01-03") or "happy" in emotion:
            return 0
        else:
            return 1