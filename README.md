# Harmoncic-percussive-residual separation for audio sources

Python CLI that generates harmonic-percussive separation with residual component on wav files using vertical and horizontal median filters.

## :notebook: Description

This application uses the signal processing theory presented in the book Fundamentals of Music Processing by Meinard Müller.

The procedure is based on the fact that harmonic components tend to be continuous in time and bounded in frequency, while percussive components are bounded in time but continuous in frequency.

The algorithm uses median filters applied horizontally and vertically to emphasize both components respectively, and includes a separation factor that allows for a more restrictive separation of harmonic and percussive elements.

Based on the filtered data, binary masks are generated to manipulate the original spectrogram and extract the information from the components for subsequent restoration using an inverse Short-Time Fourier Transform (STFT).

## :rocket: Getting Started

### :running_man: Run locally

To install this application you need to clone the repository locally and install all the dependencies specified in `requirements.txt`:

```
git clone https://github.com/m-quezada-loyola/harmonic-percussive-separation.git
```

Go to the project directory:

```
cd .\armonic-percussive-separation\
```
Install dependencies

```
pip install -r requirements.txt
```

Before running the application, make sure you have .wav files in the audios folder for processing. Once this is done, you can execute the following command:

```
python main.py
```
