import numpy as np
from keras.optimizers import SGD
from keras.models import model_from_yaml,model_from_json
from keras.preprocessing.image import ImageDataGenerator
import os
np.set_printoptions(suppress=True)

img_width=160
img_height=150



f = open('tuning_logs/2018-01-12 02-33-45/2018-01-12 02-33-45_ARCH.json', 'r')
model = model_from_json(f.read())
f.close()
model.load_weights('tuning_logs/2018-01-12 02-33-45/2018-01-12 02-33-45.hdf5')

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
datagen = ImageDataGenerator(
        rescale=1. / 255
    )

generator = datagen.flow_from_directory(
        'fma_medium_test/',
        target_size=(img_width, img_height),
        batch_size=50,
        class_mode=None,
        shuffle=False)

"""for i in enumerate(generator.filenames):
    print(i)"""

predictions=model.predict_generator(generator,steps=482)

accuracy=0
for i,n in enumerate(generator.filenames):
    if n.startswith("Classical") and np.argmax(predictions[i]) == 0:
        accuracy += 1
    elif n.startswith("Electronic") and np.argmax(predictions[i]) == 1:
        accuracy += 1
    elif n.startswith("Experimental") and np.argmax(predictions[i]) == 2:
        accuracy += 1
    elif n.startswith("Folk") and np.argmax(predictions[i]) == 3:
        accuracy += 1
    elif n.startswith("Hip-Hop") and np.argmax(predictions[i]) == 4:
        accuracy += 1
    elif n.startswith("Instrumental") and np.argmax(predictions[i]) == 5:
        accuracy += 1
    elif n.startswith("International") and np.argmax(predictions[i]) == 6:
        accuracy += 1
    elif n.startswith("Old-Time_Historic") and np.argmax(predictions[i]) == 7:
        accuracy += 1
    elif n.startswith("Pop") and np.argmax(predictions[i]) == 8:
        accuracy += 1
    elif n.startswith("Rock") and np.argmax(predictions[i]) == 9:
        accuracy += 1
print(accuracy/(np.shape(predictions)[0]))


