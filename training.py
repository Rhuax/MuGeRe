from __future__ import print_function
import os
import sys
import time
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import TensorBoard,ModelCheckpoint





batchSize = 128
epochs = 30
num_classes = 8
tb=TensorBoard(batch_size=batchSize,log_dir='./logs')#logs
model_path=datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
os.mkdir('tuning_logs/'+model_path)
checkpoint = ModelCheckpoint('tuning_logs/'+model_path+'/'+model_path+'.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


callbacks=[tb,checkpoint]
trainsetDir = 'fma_train/'
testsetDir = 'fma_test/'

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160,150), shuffle=True)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160,150))

model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=64, kernel_size=2, strides=2, activation="elu", kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2,padding='same'))

model.add(Conv2D(filters=128, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=256, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=512, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(128))

model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
opt=rmsprop()
#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#Summary
old_stdoud = sys.stdout
f=open('tuning_logs/'+model_path+'/'+model_path+'.txt','w')
sys.stdout = f
model.summary()
f.close()
sys.stdout = old_stdoud
model.summary()
f=open('tuning_logs/'+model_path+'/'+model_path+'_ARCH.json','w')
f.write(model.to_json())
f.close()


#Training
model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=8000//batchSize,
        steps_per_epoch=31985//batchSize,
        callbacks=callbacks)
