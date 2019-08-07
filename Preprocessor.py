"""
    Helper file that will perform pre-processing operations on the
    training data:
    1. Zero pad so that they are all 3 (avg length) seconds and re-sample to have 16 kHz frequency rate
    2. Add random Gaussian noise
    3. Convert to log spectogram (Hamming window = 25ms, 11.5ms overlap, DFT=512)

    :author Theodora Bendlin
"""
import numpy as np
import librosa as lib
from librosa.feature import melspectrogram
from librosa.effects import trim

# Processing parameters
RESAMPLE_RATE = 16000
SAMPLE_DURATION = 3
SAMPLE_OFFSET = 0.25
MIN_FREQUENCY = 20
MAX_FREQUENCY = 4000
SNR_POINTS = 15

'''
    Helper function that will trim or extend the signal length
    to the desired length
    
    Zero values replaced with very small values to avoid divide by zero errors
    later on in the pre-processing
    
    :parameter signal, the audio signal
    :parameter sample_rate, the sample rate (frequency) of the signal
    
    :returns signal, the audio signal
'''


def trim_signal_length(signal, sample_rate, length=SAMPLE_DURATION):
    # Trim leading and trailing white space
    signal, i = trim(y=signal)

    # Replace zero values with small value to avoid divide by zero error later on
    signal[signal == 0] = 0.0001

    # Extend audio to be 3.5 seconds long if needed
    target = int(sample_rate * length)

    signal_length = len(signal)

    # If its longer, clip the length
    if signal_length > target:
        return signal[0:target]
    else:
        return np.pad(signal, (0, target - signal_length), 'wrap')

'''
    Helper function that will add +15 SNR to the audio
    signal to introduce noise into training data (mirrors
    real environment)
    
    :parameter signal, the audio signal
    :parameter num_times, the number of times to apply the additive white gaussian noise
    
    :returns noisy_signal, the new signal with gaussian noise
'''
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

'''
    Helper function that will create a log-spectogram representing the image
    
    :parameter signal, the audio signal
    :parameter sr, the sampling rate (frequency) of the signal
    :parameter win_length, hamming window size for FFT
    :parameter overlap, the overlap size for FFT
    :parameter nnft, the number of bins for FFT
    
    :returns freqs, the frequency dimension of the spectogram
    :returns times, the time dimension of the spectogram
    :returns spectogram, the final spectogram
'''
def create_windowed_melspectrogram(audio, sr, nfft=512, overlap=0.0044, win_length=0.005):
    window_length = int(sr * win_length)
    overlap_distance = int(sr * overlap)

    spectrum = lib.stft(y=audio, n_fft=nfft, hop_length=overlap_distance, win_length=window_length, window='hamm')
    mel_spectrum = melspectrogram(S=np.abs(spectrum) ** 2, sr=sr)
    return lib.power_to_db(mel_spectrum, ref=np.max)

'''
    Helper function that will pre-process an audio signal before
    it is converted to a spectrogram

    :parameter filename, the name of the file that contains the audio sample
    :parameter noise_reps, the number of repetitions to add white gaussian noise for training

    :returns filtered_signal, the final processed audio signal
    :returns sample_rate, the sample rate (frequency) of the audio signal 
'''
def preprocess(filename, noise_reps=10):
    # Load in the file
    signal, sample_rate = lib.load(filename)

    # Re-sample the signal to 16 kHz
    resample_signal = lib.resample(signal, sample_rate, RESAMPLE_RATE)
    sample_rate = RESAMPLE_RATE

    # Trim/Extend the signal to SAMPLE_DURATION so they are all uniform length
    trimmed_signal = trim_signal_length(resample_signal, sample_rate)

    # Add noise for the number of repetitions
    noisy_signal = add_white_noise(trimmed_signal, noise_reps)

    return noisy_signal, sample_rate

'''

    Helper function that ties all pre-processing functions together
    
    :parameter filename, the name of the file that contains the audio sample
    :parameter noise_reps, the number of repetitions to add white gaussian noise for training

    :returns spectrogram, the spectrogram of the processed audio sample
'''
def preprocess_and_create_spectrogram(filename, noise_reps):
    signal, sample_rate = preprocess(filename, noise_reps)
    spectrogram = create_windowed_melspectrogram(signal, sample_rate)
    return spectrogram