from __future__ import print_function

import keras



from keras.preprocessing.image import ImageDataGenerator



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D, LeakyReLU

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.optimizers import Adadelta





import numpy as np

import os





batchSize = 128

epochs = 30

num_classes = 16

model_name = 'Genre Recognition Model'



trainsetDir = 'fma_medium_train/'

testsetDir = 'fma_medium_test/'



# Data generators

#train_datagen = ImageDataGenerator()
train_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160,150), shuffle=True)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160,150))


'''
model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=16, kernel_size=(3,3), strides=(2,2), padding='same', activation="elu", kernel_regularizer='l2', kernel_initializer='glorot_normal'))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same',activation="elu", kernel_regularizer='l2', kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same',activation="elu", kernel_regularizer='l2', kernel_initializer='glorot_normal'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same',activation="elu", kernel_regularizer='l2', kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(256, activation='elu', kernel_regularizer='l2'))
#model.add(Dropout(0.5))

model.add(Dense(128, activation='elu', kernel_regularizer='l2'))
#model.add(Dropout(0.5))

model.add(Dense(64, activation='elu', kernel_regularizer='l2'))
model.add(Dense(num_classes, activation='softmax'))
'''

#activation = 'elu'
activation = LeakyReLU()
kernel_initializer = 'glorot_normal'
kernel_regularizer = 'l2'
kernel_size = (3,3)
strides = (2,2)
optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=32, kernel_size=kernel_size, strides=strides, activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=kernel_size, strides=strides, activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=128, kernel_size=kernel_size, strides=strides,activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
model.add(Dropout(0.5))
model.add(Dense(128, activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#Compile model

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



#Summary

model.summary()



#Training

model.fit_generator(

        train_generator,

        epochs=epochs,

        validation_data=test_generator,

        validation_steps=25001//batchSize,

        steps_per_epoch=99924//batchSize)