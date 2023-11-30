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

def create_binary_masks(harmonic_spectrogram, percussive_spectrogram, beta):

    harmonic_mask   = np.int8(harmonic_spectrogram >= beta * percussive_spectrogram)
    percussive_mask = np.int8(percussive_spectrogram > beta * harmonic_spectrogram)
    residual_mask   = 1 - (harmonic_mask + percussive_mask)
    return harmonic_mask, percussive_mask, residual_mask

def apply_binary_mask(spectrogram, mask):
    
    return spectrogram * mask