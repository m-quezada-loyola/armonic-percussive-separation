from file_managment import create_audio_directory, get_audio_files
from graphics import save_plot_data
from audio_processing import generate_hprs, save_audios
import soundfile as sf
import click


@click.command()
@click.option(
    "-b", "--betas", nargs=3, required=True, type=float, show_default=True,
    help="Harmonic-percussive-residual separation factor",
)
@click.option(
    "-fl", "--frame-length", default=1024, type=int, show_default=True,
    help="Window function sample size",
)
@click.option(
    "-hs", "--hop-size", default=256, type=int, show_default=True,
    help="Number of overlap samples",
)
@click.option(
    "-hml", "--harmonic-median-length", default=0.2, type=float, show_default=True,
    help="Harmonic median filter length measured in seconds",
)
@click.option(
    "-pml", "--percussive-median-length", default=500, type=int, show_default=True,
    help="Percussive median filter length measured in hertz",
)
def main(betas, frame_length, hop_size, harmonic_median_length, percussive_median_length):
    filenames = get_audio_files("audios")
    if len(filenames):
        for filename in filenames:
            for beta in betas:
                directory_name = create_audio_directory(filename, beta)
                data, fs = sf.read(filename)
                audios, masks, spectrograms = generate_hprs(data, fs, frame_length, hop_size, beta,
                                                            harmonic_median_length, percussive_median_length)
                save_audios(directory_name, audios, fs)
                save_plot_data(directory_name, masks, "mask", hop_size, fs)
                save_plot_data(directory_name, spectrograms, "spectrogram", hop_size, fs)
    else:
        print("Audio files not found")


if __name__ == "__main__":
    main()
