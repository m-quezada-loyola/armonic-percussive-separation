import numpy as np
import librosa
import scipy.signal as signal


def calculate_spectrogram(audio_data, frame_length: int, hop_size: int):
    spectrogram = librosa.stft(audio_data, frame_length, hop_size, frame_length)
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
    harmonic_mask = np.int8(harmonic_spectrogram >= beta * percussive_spectrogram)
    percussive_mask = np.int8(percussive_spectrogram > beta * harmonic_spectrogram)
    residual_mask = 1 - (harmonic_mask + percussive_mask)
    return harmonic_mask, percussive_mask, residual_mask


def apply_binary_mask(spectrogram, mask):
    return spectrogram * mask


def recover_audio(spectrogram, frame_length, hop_size, data_length):
    audio = librosa.istft(
        spectrogram, hop_length=hop_size, win_length=frame_length, length=data_length
    )
    return audio


def calculate_frame_length(time_length, fs, hop_size):
    frame_length = int(np.ceil(time_length * fs / hop_size))
    return frame_length


def calculate_bin_length(hertz_length, fs, frame_length):
    bin_length = int(np.ceil(hertz_length * frame_length / fs))
    return bin_length


def even_to_odd(number):
    odd_number = number + 1 if (number % 2 == 0) else number
    return odd_number
