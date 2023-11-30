import numpy as np
import librosa

def calculate_spectrogram(data, frame_length: int, hop_size: int):

    spectrogram = librosa.stft(data, frame_length, hop_size, frame_length)
    return spectrogram

def calculate_power_spectrogram(spectrogram):

    return np.abs(spectrogram) ** 2