import os
import csv
from subprocess import run, PIPE

rootdir = 'fma_small\\'
dataset_mono_root = 'fma_small_final\\'

def set_to_mono(path_file, tmp_name):
    command = "C:\\Users\\Stefano\\Desktop\\prova\\sox-14-4-2\\sox.exe {} {} remix 1,2".format(path_file, tmp_name)
    run(command, shell=True, stdin=PIPE, stdout=PIPE)

# converts the audio to spectrogram
def audio_to_spect(input_file, output_file):
    command = "C:\\Users\\Stefano\\Desktop\\prova\\sox-14-4-2\\sox.exe {} -n spectrogram -r -x 800 -y 150 -o {}".format(input_file, output_file)
    run(command, shell=True, stdin=PIPE, stdout=PIPE)

# helper function to delete files no longer needed
def delete_file(file_path):
    os.remove(file_path)

def get_genre_from_track(track_name):
    reader = csv.reader(open('dataset.csv'), delimiter=",")
    for line in reader:
        if line[0] == track_name:
            return line[1]


for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))
        path_file = subdir + "\\" + file
        mono_name = dataset_mono_root + "\\" + get_genre_from_track(file[0:-4]) + "\\" + file[0:-4] + "_mono.mp3"
        new_path = dataset_mono_root + "\\" + get_genre_from_track(file[0:-4]) + "\\"
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        set_to_mono(path_file, mono_name)
        spec_file = new_path + file[0:-4] + ".png"
        audio_to_spect(mono_name, spec_file)
        delete_file(mono_name)

