import numpy as np
import librosa
import scipy.signal as signal

def calculate_spectrogram(data, frame_length: int, hop_size: int):

    spectrogram = librosa.stft(data, frame_length, hop_size, frame_length)
    return spectrogram

def calculate_power_spectrogram(spectrogram):

    return np.abs(spectrogram) ** 2

def vertical_median_filter(power_spectrogram, length):

    filtered_spectrogram = signal.medfilt(power_spectrogram, [length, 1])
    return filtered_spectrogram

def horizontal_median_filter(power_spectrogram, length):

    filtered_spectrogram = signal.medfilt(power_spectrogram, [1, length])
    return filtered_spectrogram

