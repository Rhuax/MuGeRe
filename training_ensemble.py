from __future__ import print_function
import os
import sys
import time
import datetime
from collections import Counter

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

import glob

batchSize = 128
epochs = 30
"""
num_classes = 16
nb_train_examples = 96214
nb_valid_examples = 24066
"""

trainsetDir = 'fma_medium_train/'
testsetDir = 'fma_medium_test/'

num_classes = 0
nb_train_examples = 0
nb_valid_examples = 0
for genre in sorted(os.listdir(trainsetDir)):
    if (len(glob.glob(trainsetDir + genre + '/*')) > 2000):
        nb_train_examples += len(glob.glob(trainsetDir + genre + '/*'))
        num_classes += 1

for genre in sorted(os.listdir(testsetDir)):
    if (len(glob.glob(testsetDir + genre + '/*')) > 2000):
        nb_valid_examples += len(glob.glob(testsetDir + genre + '/*'))


def calculateGenreWeight():
    weights = np.zeros(num_classes)
    i = 0
    for genre in sorted(os.listdir(trainsetDir)):
        weights[i] = len(glob.glob(trainsetDir + genre + '/*'))
        i += 1
    proata = dict(zip(range(num_classes), np.amax(weights) / weights))
    return proata


tb = TensorBoard(batch_size=batchSize, log_dir='./logs')  # logs
model_path = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
os.mkdir('tuning_logs/' + model_path)
checkpoint = ModelCheckpoint('tuning_logs/' + model_path + '/' + model_path + '.hdf5', monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')

callbacks = [tb, checkpoint]

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160, 150),
                                                    shuffle=False)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160, 150))

model = keras.applications.resnet50.ResNet50(include_top=True, weights='None', input_tensor=None,
                                             input_shape=(160, 150, 3), pooling=None, classes=num_classes)

"""
Successivamente aggiungiamo altri eventuali modelli
"""

"""
model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=64, kernel_size=(3, 3), strides=(3, 3), activation="elu",
                 kernel_initializer='glorot_normal'))
model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), activation="elu", kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=16, kernel_size=(2, 2), strides=(2, 2), activation="elu", kernel_initializer='glorot_normal'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_normal'))
model.add(Dense(num_classes, activation='softmax'))

"""

# Compile model

optimizer = 'adam'

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
# Summary

model.summary()
old_stdoud = sys.stdout
f = open('tuning_logs/' + model_path + '/' + model_path + '.txt', 'w')
sys.stdout = f
model.summary()
f.close()
sys.stdout = old_stdoud
f = open('tuning_logs/' + model_path + '/' + model_path + '_ARCH.json', 'w')
f.write(model.to_json())
f.close()

# Training
model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=8000 // batchSize,
    steps_per_epoch=nb_train_examples // batchSize,
    callbacks=callbacks,
    class_weight=calculateGenreWeight())
