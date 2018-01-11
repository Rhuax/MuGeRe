from keras.optimizers import SGD
from keras.models import model_from_yaml,model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from PIL import Image
from subprocess import run,PIPE
import os
np.set_printoptions(suppress=True)

img_width=160
img_height=150

arch_path="arch_weights/MuGeRe Architecture_T7.yaml"
weights_path="arch_weights/MuGeRe Weights_T7.h5"
song_path="bach.mp3"

datagen = ImageDataGenerator(
        rescale=1. / 255
    )

is_temp=False

def stereo2mono(a):
    c='sox '+a+' temp.mp3 remix 1,2'
    run(c, shell=True, stdin=PIPE, stdout=PIPE)
def audio2spec(name='temp.mp3'):
    c='sox '+name+' -n spectrogram -r -x 800 -y 150 -o temp.png'
    run(c,shell=True,stdin=PIPE, stdout=PIPE)
"""
Split original spectrogram in chunks wide 160px
"""
def split_chunks(base_name):
    if not os.path.exists('prediction_temp'):
        os.mkdir('prediction_temp')
    for j in range(0,800,160):
        i=Image.open('temp.png')
        box=(j,0,j+160,150)
        crop=i.crop(box)
        crop.save('prediction_temp/as/'+base_name+'_'+str(j//160)+'.png')
    os.remove('temp.png')


#Load the network
model=None
f=open(arch_path,'r')
print("Loading architecture...",end='')
if arch_path[-4:]=='yaml':
    model=model_from_yaml(f.read())
elif arch_path[-4:]=='json':
    model=model_from_json(f.read())
else:
    print("Couldn't load network architecture. Bad file path")
    exit()
print("loaded")
print("Loading network weights...",end='')
model.load_weights(weights_path)
print("loaded")
print("Compiling...",end='')
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
print("compiled")

try:
    audio = AudioSegment.from_mp3(os.path.abspath(song_path))
    if audio.channels==2:
        is_temp=True
        print("Converting mono to stereo..",end='')
        stereo2mono(os.path.abspath(song_path))
        print("converted")
except CouldntDecodeError:
    print("Found a bad mp3, exiting.")
    exit()
if is_temp:
    audio2spec()
    os.remove('temp.mp3')
else:
    audio2spec(song_path)

split_chunks(song_path)
#Prediction
generator = datagen.flow_from_directory(
        'prediction_temp/',
        target_size=(img_width, img_height),
        batch_size=5,
        class_mode=None,
        shuffle=False)

predictions=model.predict_generator(generator,steps=1)
print(predictions)