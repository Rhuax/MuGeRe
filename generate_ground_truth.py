"""
This file uses fma_small directory and tracks.csv to generate a single csv file containing:
id_track, genre

It is used as ground truth in the learning process.
"""

import csv
import glob

def getTopgenre(id_subgenre):
    reader = csv.reader(open('genres.csv'), delimiter=",")
    reader2 = csv.reader(open('genres.csv'), delimiter=",")

    for line in reader:
        if line[0]==id_subgenre:
            for l in reader2:
                if l[0]==line[-1]:
                    return l[-2]





def extract_trackid_topgenres():
    reader=csv.reader(open('tracks.csv'),delimiter=",")
    #13 genre top
    i=0
    for line in reader:
        i+=1
        if i==3:
            break
    file=open('training_set.csv','w')
    dict={'Hip-Hop':0,'Pop':0,'Rock':0,'Experimental':0,'Folk':0,'Jazz':0,'Electronic':0,'Spoken':0,
          'International':0,'Soul-RnB':0,'Blues':0,'Country':0,'Classical':0,'Old-Time / Historic':0,
          'Instrumental':0,'Easy Listening':0}
    #writer.writerow(['track_id','genre'])
    for line in reader:
        genre=line[-13]
        track_id=line[0]
        if(genre==''):
            #Retrieve its top genre
            sub=line[-12]
            sub=sub[1:-1]
            sub=sub.split(',')
            genre=getTopgenre(sub[0])
        if genre is not None and genre!='':
            file.write(track_id+','+genre+'\n')
            dict[genre]=dict[genre]+1

    for k in dict.keys():

        print(k,end='')
        print(' ',end='')
        print(dict[k])

    print(dict)
    file.close()

"""Generate the effective training_set since not all metadata tracks are in the
small version of fma dataset"""
def generate_true_dataset():
    reader = csv.reader(open('training_set.csv'), delimiter=",")
    file=open('dataset.csv','w')
    for line in reader:
        id=line[0].zfill(6) #Pad with zeros since mp3 files are padded with zeros -_-"

        #Check if a file named id.mp3 is present in one of all sub-folders
        for f in glob.glob('fma_small/*/'+id+'.mp3',recursive=True):
            #I'm inside the loop so there's a file
            file.write(id+','+line[1]+'\n')

    file.close()
def count_genres(file):
    reader = csv.reader(open(file), delimiter=",")
    dict = {'Hip-Hop': 0, 'Pop': 0, 'Rock': 0, 'Experimental': 0, 'Folk': 0, 'Jazz': 0, 'Electronic': 0, 'Spoken': 0,
            'International': 0, 'Soul-RnB': 0, 'Blues': 0, 'Country': 0, 'Classical': 0, 'Old-Time / Historic': 0,
            'Instrumental': 0, 'Easy Listening': 0}
    for line in reader:
        dict[line[1]]=dict[line[1]]+1
    for k in dict.keys():
        print(k+' '+str(dict[k]))

count_genres('dataset.csv')