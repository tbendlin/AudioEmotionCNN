"""
    Helper file that will perform pre-processing operations on the
    training data:
    1. Zero pad so that they are all 3.5 (avg length) seconds and re-sample to have 16 kHz frequency rate
    2. Pass through lowpass FIR filter
    3. Convert to log spectogram (Hamming window = 25ms, 11.5ms overlap, DFT=512)
    4. Cut the size in half
    5. Z-normalize to have zero mean and standard deviation close to one

    :author Theodora Bendlin
"""
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import librosa as lib
import cv2 as cv

# Processing parameters
RESAMPLE_RATE = 16000
SAMPLE_DURATION = 3.5
MAX_FREQUENCY = 4000
SNR_POINTS = 15

'''
    Helper function that will pre-process an audio signal before
    it is converted to a spectogram
    
    :parameter filename, the name of the file that contains the audio sample
    :parameter noise_reps, the number of repetitions to add white gaussian noise for training
    
    :returns filtered_signal, the final processed audio signal
    :returns sample_rate, the sample rate (frequency) of the audio signal 
'''
def preprocess(filename, noise_reps=10):

    # Load in the file
    signal, sample_rate = lib.load(filename)

    # Trim/Extend the signal to SAMPLE_DURATION so they are all uniform length
    trimmed_signal = trim_signal_length(signal, sample_rate)

    # Re-sample the signal to 16 kHz
    resample_signal = lib.resample(trimmed_signal, sample_rate, RESAMPLE_RATE)

    noisy_signal = add_white_noise(resample_signal, noise_reps)

    # Pass through lowband FIR filter
    filtered_signal = fir_filter(noisy_signal)

    return filtered_signal, sample_rate

'''
    Helper function that will trim or extend the signal length
    to the desired length
    
    Zero values replaced with very small values to avoid divide by zero errors
    later on in the pre-processing
    
    :parameter signal, the audio signal
    :parameter sample_rate, the sample rate (frequency) of the signal
    
    :returns signal, the audio signal
'''
def trim_signal_length(signal, sample_rate):

    # Replace zero values with small value to avoid divide by zero error later on
    signal[signal == 0] = 0.0001

    # Extend audio to be 3.5 seconds long if needed
    target = int(sample_rate * SAMPLE_DURATION)
    if len(signal) < target:
        padding = target - signal.shape[0]
        if padding > 0:
            signal = np.pad(signal, (0, padding), mode='constant', constant_values=(0.0001))
    else:
        signal = signal[:target]

    return signal

'''
    Helper function that passes the signal through an FIR
    lowpass filter
    
    :parameter signal, the audio signal
    
    :returns signal, the audio signal
'''
def fir_filter(signal):
    out = sig.butter(1, Wn=0.35, btype='lowpass', output='ba')
    b, a = out[0], out[1]
    return sig.lfilter(b, a, signal)

'''
    Helper function that will add +15 SNR to the audio
    signal to introduce noise into training data (mirrors
    real environment)
    
    :parameter signal, the audio signal
    :parameter num_times, the number of times to apply the additive white gaussian noise
    
    :returns noisy_signal, the new signal with gaussian noise
'''
def add_white_noise(signal, num_times):

    noisy_signal = signal
    for _ in range(num_times):
        noise = np.random.rand(len(signal))
        Ps = np.sum(signal ** 2) / len(signal)
        Pn = np.sum(noise ** 2) / len(signal)

        scale_factor = (Ps / Pn) / SNR_POINTS
        gaussian_noise = noise * scale_factor

        noisy_signal = noisy_signal + gaussian_noise

    return noisy_signal

'''
    Helper function that will create a log-spectogram representing the image
    
    :parameter signal, the audio signal
    :parameter sample_rate, the sampling rate (frequency) of the signal
    :parameter window_size, hamming window size for FFT
    :parameter overlap_size, the overlap size for FFT
    :parameter nnft, the number of bins for FFT
    
    :returns freqs, the frequency dimension of the spectogram
    :returns times, the time dimension of the spectogram
    :returns spectogram, the final spectogram
'''
def create_spectogram(signal, sample_rate=RESAMPLE_RATE, window_size=0.025, overlap_size=0.0115, nnft=512):
    nperseg = int(round(window_size * sample_rate))
    noverlap = int(round(overlap_size * sample_rate))
    freqs, times, spec =  sig.spectrogram(signal,
                                          fs=sample_rate,
                                          window='hamming',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          nfft=nnft)
    return freqs, times, np.log(spec.T.astype(np.float32))

'''
    Helper function that will resize the spectogram to half its dimensions for the CNN
    
    :paramter spectogram, the spectogram to be resized
    
    :returns spectogram, the resized spectogram
'''
def resize_spectogram(spectogram):
    x_dim = int(spectogram.shape[0] * 0.5)
    y_dim = int(spectogram.shape[1] * 0.5)
    return cv.resize(spectogram, (x_dim, y_dim))

'''
    Helper function that will normalize the spectogram to have a mean
    and standard deviation close to one
    
    :parameter spectogram, the spectogram to normalize
    
    :returns spectogram, the normalized spectogram
'''
def normalize_spectorgram(spectogram):
    mean = spectogram.mean()
    dev = spectogram.std()
    znorm_func = np.vectorize(znorm)

    znorm_func(spectogram, mean, dev)

    return spectogram

def znorm(xi, mean, dev):
    return (xi - mean) / dev

'''
    Helper function that will plot a spectogram for debugging purposes
    
    :parameter spectogram, the final spectogram
'''
def plot_spectogram(spectogram):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(212)
    ax1.imshow(spectogram.T, aspect='auto', origin='lower',
               extent=[0, len(spectogram), 0, len(spectogram[0])])
    ax1.set_title('Spectrogram')
    ax1.set_ylabel('Freqs in Hz')
    ax1.set_xlabel('Seconds')

    plt.show()