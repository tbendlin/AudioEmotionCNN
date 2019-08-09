"""
    File for processign audio, same as what was used for making
    the training data for the neural network
"""

# All pre-processing function definitions

import time
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
import librosa as lib
from librosa.feature import melspectrogram
from librosa.filters import mel
from librosa.util import normalize

# Processing Hyperparameters

RESAMPLE_RATE = 16000
SAMPLE_DURATION = 3
SAMPLE_OFFSET = 0.25
MIN_FREQUENCY = 20
MAX_FREQUENCY = 4000
SNR_POINTS = 15

# Trim the length of the signal
def trim_signal_length(signal, sample_rate, length=SAMPLE_DURATION):

  # Replace zero values with small value to avoid divide by zero error later on
  signal[signal == 0] = 0.0001

  # Extend audio to be 3.5 seconds long if needed
  target = int(sample_rate * SAMPLE_DURATION)

  signal_length = len(signal)

  # If its longer, clip the length
  if signal_length > target:
    return signal[0:target]
  else:
    return np.pad(signal, (0, target - signal_length), 'wrap')

  return signal

# Add white noise to the signal for a specific number of repetitions
def add_white_noise(signal, noise_reps):
    Ps_avg = np.sum(signal ** 2) / len(signal)
    Ps_avg_db = 10 * np.log10(Ps_avg)

    Pn_avg_db = Ps_avg_db - SNR_POINTS
    Pn_avg = 10 ** (Pn_avg_db / 10)

    noisy_signal = signal

    for _ in range(noise_reps):
      noise = np.random.normal(loc=0, scale=np.sqrt(Pn_avg), size=len(signal))
      noisy_signal = noisy_signal + noise
    
    return noisy_signal

# Applies the previous pre-processing operations
def preprocess(filename, noise_reps=10):
    
    start = time.time()
    # Load in the file
    signal, sample_rate = lib.load(filename)
    end = time.time()
    print("Sample loaded in: ", end - start)
    
    start = time.time()
    # Re-sample the signal to 16 kHz
    resample_signal = lib.resample(signal, sample_rate, RESAMPLE_RATE)
    sample_rate = RESAMPLE_RATE
    end = time.time()
    print("Resampled in: ", end - start)
    
    start = time.time()
    trimmed_signal = trim_signal_length(resample_signal, sample_rate)
    end = time.time()
    
    print("Trimmed in: ", end - start)

    start = time.time()
    # Add noise for the number of repetitions
    noisy_signal = add_white_noise(trimmed_signal, noise_reps)
    end = time.time()
    
    print("Noise added in: ", end - start)

    return noisy_signal, sample_rate

# Trying out a mel-spectogram where more params are specified
def create_windowed_melspectrogram(audio, sr, nfft=512, overlap=64,
                                    win_length=128):
  freq, time, spectrum = sig.stft(x=audio, fs=sr, nfft=nfft, nperseg=win_length, window='hamming', noverlap=overlap)
  mel_spectrum = melspectrogram(S=np.abs(spectrum)**2, sr=sr)
  power_spectrum = lib.power_to_db(mel_spectrum, ref=np.max)
  return power_spectrum.astype(np.float32)

# Function tying in all operations for creating the spectrogram image
def preprocess_and_create_spectrogram(filename, noise_reps):
    signal, sample_rate = preprocess(filename, noise_reps)
    spectrogram = create_windowed_melspectrogram(signal, sample_rate)
    return spectrogram