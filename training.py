from __future__ import print_function
import sys
import time
import datetime

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import TensorBoard




batchSize = 128
epochs = 30
num_classes = 8
tb=TensorBoard(batch_size=batchSize,log_dir='./logs')#logs


callbacks=[tb]
trainsetDir = 'fma_train/'
testsetDir = 'fma_test/'

# Data generators
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(directory=trainsetDir, batch_size=batchSize, target_size=(160,150), shuffle=True)
test_generator = test_datagen.flow_from_directory(directory=testsetDir, batch_size=batchSize, target_size=(160,150))

model = Sequential()

model.add(Conv2D(input_shape=(160, 150, 3), filters=32, kernel_size=(2,2), strides=2, activation="elu", kernel_regularizer='l2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(2,2), strides=2, activation="elu", kernel_regularizer='l2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(2,2), strides=2, activation="elu", kernel_regularizer='l2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(Flatten())
model.add(Dense(512, activation='elu', kernel_regularizer='l2'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation='elu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Summary
old_stdoud = sys.stdout
f=open('tuning_logs/'+str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')),'w')
sys.stdout = f
print(model.summary())
f.close()
sys.stdout = old_stdoud


#Training
model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=8000//batchSize,
        steps_per_epoch=31985//batchSize,
        callbacks=callbacks)
