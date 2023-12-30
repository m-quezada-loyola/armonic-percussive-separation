import matplotlib.pyplot as plt
import librosa
import numpy as np
from os.path import join
    

def save_plot_data(directory_name, dict_data, data_name, hop_size, fs):
    for key, value in dict_data.items():
        title = f'{key} {data_name}'.capitalize()
        filename = join(directory_name, f'{key}_{data_name}.png')
        create_plot(value, title, directory_name, filename, hop_size, fs)
    return None


def create_plot(data, title, directory_name, filename, hop_size, fs):
    fig, ax = plt.subplots(figsize=(12,9))
    S_db_hr = librosa.amplitude_to_db(np.abs(data), ref=np.max)
    img = librosa.display.specshow(S_db_hr, hop_length=hop_size, sr=fs, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.savefig(filename, dpi=250)
    plt.close()
    return None