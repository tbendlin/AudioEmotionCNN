"""
    File containing custom implementation for a PyTorch Dataset

    :author Theodora Bendlin
"""

import Preprocessor as pre
from torch.utils.data import Dataset
from torch import from_numpy
from os import listdir

'''
    Custom implementation of an Audio Dataset that will load in the 
    RAVDESS and SAVEE audio samples and transform them into PyTorch
    Tensors to be used as input into the CNN
    
    Performs the following transformations to create data tensor:
    1. Loads in and pre-processes audio sample
    2. Creates a spectogram from the processed audio sample
    3. Resizes the spectogram to be 128x127
    4. Z-normalizes the spectogram to have a mean and std close to 1
    5. Convert numpy array to PyTorch tensor
'''
class AudioDataset(Dataset):
    def __init__(self, data_root, is_test):
        self.samples = []
        self.labels = []

        # Only adding Gaussian noise for training data set
        noise_reps = 10
        if is_test:
            noise_reps = 0

        max = 20
        for voice_sample in listdir(data_root):
            if voice_sample.startswith("."):
                continue

            if max == 0:
                break

            signal, sample_rate = pre.preprocess(data_root + voice_sample, noise_reps)
            _, _, spectogram = pre.create_spectogram(signal)
            spectogram = pre.resize_spectogram(spectogram)
            spectogram = pre.normalize_spectorgram(spectogram)
            label = self.__getclasslabel__(voice_sample)

            self.samples.append(from_numpy(spectogram))
            self.labels.append(label)

            max -= 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
       return self.samples[idx], self.labels[idx]

    '''
        Helper function that will translate the file names to label
        Assume filename is in the form src--emotion-codec.wav
    '''
    def __getclasslabel__(self, file_name):
        RAVDESS_emotions = ["03-01-01", "03-01-03", "03-01-04", "03-01-05", "03-01-06"]
        SAVEE_emotions = ["n", "h", "sa", "a", "f"]

        name_parts = file_name.split("--")
        emotion = name_parts[len(name_parts) - 1]

        for idx in range(len(RAVDESS_emotions)):
            if emotion.startswith(RAVDESS_emotions[idx]):
                return idx

        for idx in range(len(SAVEE_emotions)):
            if emotion.startswith(SAVEE_emotions[idx]):
                return idx

        return 0
