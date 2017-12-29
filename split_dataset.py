import os
from shutil import copyfile
'''
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
'''
'''
dataset='fma_medium_final/'
trainset = 'fma_medium_train/'
testset = 'fma_medium_test/'
category = ''
n = 0
for dirs in os.listdir(dataset):
    n+=1
    category = ''
    length = 0

    os.mkdir(trainset + dirs)
    os.mkdir(testset + dirs)

    for files in os.listdir(dataset + dirs):
        length+=1

    train_length = (length/100) * 80
    test_length = (length/100) * 20

    index = 0

    for files in os.listdir(dataset + dirs):
        index+=1
        if(index<train_length):
            copyfile(dataset + dirs + '/' + files, trainset + dirs + '/' + files)
        else:
            copyfile(dataset + dirs + '/' + files, testset + dirs + '/' + files)


    #category += dirs + '___Tot: ' + str(int(length))+ '---> ' + ' Train: '+str(int(train_length))+ ' Test: '+str(int(test_length)) + '\n'
    #print(category)

    print('Folder ' + str(n) + ' completed! \n')
'''






