from __future__ import print_function
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D


import numpy as np
import os


batchSize = 32
epochs = 50
num_classes = 8
model_name = 'Genre Recognition Model'

datasetDir = 'fma_final/'
testsetDir = 'fma_final_test/'

# Data generators
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(directory=datasetDir, batch_size=batchSize, target_size=(160,150))
#test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160,150))

model = Sequential()
#(CONV(relu) + MAXPOOL) * 3 + FC * 2

model.add(Conv2D(input_shape=(160, 150, 3), filters=64, kernel_size=(4,4), strides=(2,2), activation="relu"))
#ConvLayer_1 output dimensions:
#   W = (160-4)/2 + 1 = 79
#   H = (150-4)/2 + 1 = 74
#   D = 64

model.add(MaxPooling2D(pool_size=(3, 2), strides=(2,2))) #overlapping pooling
#MaxPoolLayer_1 output dimensions:
#   W = (79-3)/2 + 1 = 39
#   H = (74-2)/2 + 1 = 37
#   D = 64

model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
#ConvLayer_2 output dimensions:
#   W = (39-3)/2 + 1 = 19
#   H = (37-3)/2 + 1 = 18
#   D = 128

model.add(MaxPooling2D(pool_size=(3, 2), strides=(2,2))) #overlapping pooling
#MaxPoolLayer_2 output dimensions:
#   W = (19-3)/2 + 1 = 9
#   H = (18-2)/2 + 1 = 9
#   D = 128

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation="relu"))
#ConvLayer_3 output dimensions:
#   W = (9-3)/1 + 1 = 7
#   H = (9-3)/1 + 1 = 7
#   D = 256

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
#MaxPoolLayer_3 output dimensions:
#   W = (7-2)/1 + 1 = 6
#   H = (7-2)/1 + 1 = 6
#   D = 256

model.add(Dense(units=256, activation='sigmoid'))
#FCLayer_1 output dimensions:
#   dim = 256
#   weights = 6*6*256*256 = 2,359,296

model.add(Dense(units=8, activation='sigmoid'))
#FCLayer_1 output dimensions:
#   dim = 8
#   weights = 256*8 = 2048


#RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#Training
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

