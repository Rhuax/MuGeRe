import numpy as np
from keras.optimizers import SGD
from keras.models import model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
import os
np.set_printoptions(suppress=True)

img_width=160
img_height=150



f = open('tuning_logs/2018-01-10 16-27-57/2018-01-10 16-27-57_ARCH.json', 'r')
model = model_from_yaml(f.read())
f.close()
model.load_weights('tuning_logs/2018-01-10 16-27-57/2018-01-10 16-27-57.hdf5')

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
predictions=model.predict_generator(generator,steps=501)
all_pred = np.argmax(predictions, axis=1)
confusion_matrix = np.zeros((16, 16))
for i in range(np.size(all_pred)):
    column=-1
    if i<75:#it's blues
        column=0
    elif i>=75 and i<620+75:# it's classical
        column=1
    elif i>=620+75 and i<620+75+179:#country
        column=2
    elif i>=620+75+179 and i<620+75+179+22:#easy
        column=3
    elif i>=620+75+179+22 and i<620+75+179+22+6312:#elec
        column=4
    elif i >= 620 + 75 + 179 + 22+6312 and i < 620 + 75 + 179 + 22 + 6312+2251:  # elec
        column = 5
    elif i >= 620 + 75 + 179 + 22 + 6312+2251 and i < 620 + 75 + 179 + 22 + 6312 + 2251+1519:  # elec
        column = 6
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 +1519 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519+2198:  # elec
        column = 7
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 +1519+2198 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519+2198+1350:  # elec
        column = 8
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198+1350 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350+1019:  # elec
        column = 9
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350+1019 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019+385:  # elec
        column = 10
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019+385 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385+511:  # elec
        column = 11
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 +511 and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511+1187:  # elec
        column = 12
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 +1187and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 + 1187+7099:  # elec
        column = 13
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 + 1187 +7099and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 + 1187 + 7099+155:  # elec
        column = 14
    elif i >= 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 + 1187 + 7099 +155and i < 620 + 75 + 179 + 22 + 6312 + 2251 + 1519 + 2198 + 1350 + 1019 + 385 + 511 + 1187 + 7099 + 155+119:  # elec
        column = 15

    confusion_matrix[all_pred[i]][column] += 1


classes=sorted(os.listdir('fma_medium_train'))
for i in range(16):
    print(classes[i],end=' ')
    print(confusion_matrix[i][i]/np.sum(confusion_matrix[:,i]))

print(confusion_matrix)
