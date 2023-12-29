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


def generate_hprs(data, fs, frame_length, hop_size, beta, harmonic_median_length, percussive_median_length):

    median_length_frames = even_to_odd(calculate_frame_length(harmonic_median_length, fs, hop_size))
    median_length_bins = even_to_odd(calculate_bin_length(percussive_median_length, fs, frame_length))

    spectrogram = calculate_spectrogram(data, frame_length, hop_size)
    power_spectrogram = calculate_power_spectrogram(spectrogram)
    harmonic_power_spectrogram = horizontal_median_filter(power_spectrogram, median_length_frames)
    percussive_power_spectrogram = vertical_median_filter(power_spectrogram, median_length_bins)

    harmonic_mask = create_harmonic_mask(harmonic_power_spectrogram, percussive_power_spectrogram, beta)
    percussive_mask = create_percussive_mask(harmonic_power_spectrogram, percussive_power_spectrogram, beta)
    residual_mask = create_residual_mask(harmonic_mask, percussive_mask)

    harmonic_audio, harmonic_spectrogram = generate_audio_component(spectrogram, harmonic_mask, frame_length, hop_size, data.size)
    percussive_audio, percussive_spectrogram = generate_audio_component(spectrogram, percussive_mask, frame_length, hop_size, data.size)
    residual_audio, residual_spectrogram = generate_audio_component(spectrogram, residual_mask, frame_length, hop_size, data.size)

    audios = {
        'harmonic': harmonic_audio, 'percussive': percussive_audio, 'residual': residual_audio
    }

    masks = {
        'harmonic': harmonic_mask, 'percussive': percussive_mask, 'residual': residual_mask
    }

    spectrograms = {
        'original': spectrogram, 'harmonic': harmonic_spectrogram,
        'percussive': percussive_spectrogram, 'residual': residual_spectrogram
    }

    return audios, masks, spectrograms