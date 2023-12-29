from os import listdir, makedirs
from os.path import isfile, isdir, join

def get_audio_files(audio_folder):
    dirs = listdir(audio_folder)
    files = [join(audio_folder, f) for f in dirs if isfile(join(audio_folder, f))]
    audio_files = [f for f in files if f.endswith('.wav')]
    return audio_files

def create_audio_directory(filename, beta):
    directory_name = join(filename.split(".")[0], str(beta))
    if not isdir(directory_name):
        makedirs(directory_name)
    return directory_name
