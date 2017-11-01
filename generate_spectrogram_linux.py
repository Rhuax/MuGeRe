import glob
from shutil import copyfile
import csv
import os
from subprocess import run, PIPE
from PIL import Image
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
dataset='fma_small/'
final_dataset='fma_final/'

def stereo2mono(a):
    c='sox '+a+' temp.mp3 remix 1,2'
    run(c, shell=True, stdin=PIPE, stdout=PIPE)
def audio2spec():
    c='sox temp.mp3 -n spectrogram -r -x 800 -y 150 -o temp.png'
    run(c,shell=True,stdin=PIPE, stdout=PIPE)

def split_chunks(base_name):
    for j in range(0,800,160):
        i=Image.open('temp.png')
        box=(j,0,j+160,150)
        crop=i.crop(box)
        crop.save(base_name+'_'+str(j//160)+'.png')

with open('dataset.csv',mode='r') as f:
    reader=csv.reader(f)
    dict={r[0]:r[1] for r in reader}

if not os.path.exists(final_dataset):
    os.mkdir(final_dataset)
i=0
mono=0
prev=-1
for f in glob.glob(dataset+'*/*.mp3'):
    genre=dict[os.path.basename(f)[0:-4]]
    if not os.path.exists(final_dataset+genre):
        os.mkdir(final_dataset+genre)
    name=os.path.basename(f)[0:-4]
    name+='_0.png'
    found=False
    for d,s,fii in os.walk(final_dataset):
        if name in fii:
            found=True
            print("Salto")
            break
    if found==False:
        try:
            audio = AudioSegment.from_mp3(os.path.abspath(f))
            if audio.channels==2:
                stereo2mono(os.path.abspath(f))
            else:
                copyfile(os.path.abspath(f),'temp.mp3')

                mono+=1

            audio2spec()
            split_chunks(final_dataset+genre+'/'+os.path.basename(f)[0:-4])

        except CouldntDecodeError:
            print("Un mp3 fa cagare al cazzo")
    i += 1
    a = (i / 8000) * 100
    a = int(a)
    if a != prev:
        print(str(a) + '%')
        prev = a


print("Trovati "+str(mono)+" file mono -.-'")
