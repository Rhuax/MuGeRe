import os
from shutil import copyfile

dataset='fma_final/'
trainset = 'fma_train/'
testset = 'fma_test/'


for dirs in os.listdir(dataset):
    index = 0
    os.mkdir(trainset+dirs)
    os.mkdir(testset+dirs)
    for files in os.listdir(dataset+dirs):
        if (index < 1000):
            copyfile(dataset+dirs + '/' + files, testset+dirs+'/'+files)
            index+=1
        else:
            copyfile(dataset + dirs + '/' + files, trainset + dirs+'/'+files)
            index += 1







