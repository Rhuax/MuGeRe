from __future__ import print_function
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D


import numpy as np
import os


batchSize = 32
epochs = 50
num_classes = 8
model_name = 'Genre Recognition Model'

trainsetDir = 'fma_train/'
testsetDir = 'fma_test/'

# Data generators
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160,150), shuffle=True)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160,150))

model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=32, kernel_size=(2,2), strides=2, activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(2,2), strides=2, activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(2,2), strides=2, activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(256, activation='elu'))
#model.add(Dropout(0.25))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Summary
model.summary()

#Training
model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=test_generator,
        validation_steps=8000//batchSize,
        steps_per_epoch=31985//batchSize)
