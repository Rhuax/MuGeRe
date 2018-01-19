
from __future__ import print_function
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D, LeakyReLU, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adadelta, sgd





import numpy as np

import os





batchSize = 128

epochs = 12

num_classes = 10

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

#kernel_regularizer='l2', kernel_initializer='glorot_normal'
'''
model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=32, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(2,2),activation="elu", kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())

model.add(Dense(512, activation='elu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='elu', kernel_initializer='glorot_normal'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
'''

model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=32, kernel_size=(2,15), strides=(2,15), activation="elu", kernel_initializer='glorot_normal'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(3,3),activation="elu", kernel_initializer='glorot_normal'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(filters=8, kernel_size=(2,2), strides=(2,2), activation="elu", kernel_initializer='glorot_normal'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu', kernel_initializer='glorot_normal'))
model.add(Dense(num_classes, activation='softmax'))


#Compile model

optimizer = 'adam'
#optimizer = sgd(lr=0.01, momentum=0.8, decay=0.0, nesterov=True)
model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



#Summary

model.summary()



#Training
#model.load_weights('arch_weights/MuGeRe Weights_F1.h5', by_name=True)
model.fit_generator(

        train_generator,

        epochs=epochs,

        validation_data=test_generator,

        validation_steps=24066//batchSize,

        steps_per_epoch=96214//batchSize)


# serialize model to YAML
model_yaml = model.to_yaml()
with open("MuGeRe Architecture_F.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("MuGeRe Weights_F.h5")
print("Saved model to disk")