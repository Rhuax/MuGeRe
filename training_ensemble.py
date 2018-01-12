from __future__ import print_function

import datetime
import glob
import os
import sys
import time

import numpy as np
from keras import Model, Input
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import AveragePooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Convolution2D, merge,Conv2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator

batchSize = 16
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
model_path='2018-01-12 02-33-45'
#os.mkdir('tuning_logs/' + model_path)
checkpoint = ModelCheckpoint('tuning_logs/' + model_path + '/' + model_path + '.hdf5', monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')

callbacks = [tb, checkpoint]

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160, 150),
                                                    shuffle=True)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160, 150))

"""
model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None,
                                             input_shape=(160, 150, 3), pooling=None, classes=num_classes)
"""

"""
Successivamente aggiungiamo altri eventuali modelli
"""


def relu(x):
    return Activation('relu')(x)


def neck(nip, nop, stride):
    def unit(x):
        nBottleneckPlane = int(nop / 4)
        nbp = nBottleneckPlane

        if nip == nop:
            ident = x

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution2D(nbp, 1, 1,
                              subsample=(stride, stride))(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution2D(nbp, 3, 3, border_mode='same')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution2D(nop, 1, 1)(x)

            out = merge([ident, x], mode='sum')
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x

            x = Convolution2D(nbp, 1, 1,
                              subsample=(stride, stride))(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution2D(nbp, 3, 3, border_mode='same')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Convolution2D(nop, 1, 1)(x)

            ident = Convolution2D(nop, 1, 1,
                                  subsample=(stride, stride))(ident)

            out = merge([ident, x], mode='sum')

        return out

    return unit


def cake(nip, nop, layers, std):
    def unit(x):
        for i in range(layers):
            if i == 0:
                x = neck(nip, nop, std)(x)
            else:
                x = neck(nop, nop, 1)(x)
        return x

    return unit

inp = Input(shape=(160,150,3))
i = inp

i = Convolution2D(16,3,3,border_mode='same')(i)

i = cake(16, 32, 3, 1)(inp)  # 32x32
i = cake(32, 64, 3, 2)(i)  # 16x16
i = cake(64, 128, 3, 2)(i)  # 8x8

i = BatchNormalization(axis=-1)(i)
i = relu(i)

i = AveragePooling2D(pool_size=(8, 8), border_mode='valid')(i)  # 1x1
i = Flatten()(i)  # 128

i = Dense(10)(i)
i = Activation('softmax')(i)

model = Model(input=inp, output=i)


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

optimizer = Adam()
model.load_weights('tuning_logs/2018-01-12 02-33-45/2018-01-12 02-33-45.hdf5')
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
