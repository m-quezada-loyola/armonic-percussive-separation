import numpy as np
import librosa
import scipy.signal as signal


def calculate_spectrogram(audio_data, frame_length: int, hop_size: int):
    return librosa.stft(audio_data, frame_length, hop_size, frame_length)


def calculate_power_spectrogram(spectrogram):
    return np.abs(spectrogram) ** 2


def vertical_median_filter(power_spectrogram, length):
    return signal.medfilt(power_spectrogram, [length, 1])


def horizontal_median_filter(power_spectrogram, length):
    return signal.medfilt(power_spectrogram, [1, length])


def create_harmonic_mask(harmonic_power_spectrogram, percussive_power_spectrogram, beta):
    return np.int8(harmonic_power_spectrogram >= beta * percussive_power_spectrogram)


def create_percussive_mask(harmonic_power_spectrogram, percussive_power_spectrogram, beta):
    return np.int8(harmonic_power_spectrogram > beta * percussive_power_spectrogram)


def create_residual_mask(harmonic_mask, percussive_mask):
    return 1 - (harmonic_mask + percussive_mask)

def apply_binary_mask(spectrogram, mask):
    return spectrogram * mask


def recover_audio(spectrogram, frame_length, hop_size, data_length):
    return librosa.istft(
        spectrogram, hop_length=hop_size, win_length=frame_length, length=data_length
    )


def calculate_frame_length(time_length, fs, hop_size):
    return int(np.ceil(time_length * fs / hop_size))


def calculate_bin_length(hertz_length, fs, frame_length):
    return int(np.ceil(hertz_length * frame_length / fs))


def even_to_odd(number):
    return number + 1 if (number % 2 == 0) else number

def generate_audio_component(spectrogram, component_mask, frame_length, hop_size, data_length):
    component_spectrogram = apply_binary_mask(spectrogram, component_mask)
    audio_component = recover_audio(component_spectrogram, frame_length, hop_size, data_length)
    return audio_component, component_spectrogram

