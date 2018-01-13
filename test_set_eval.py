import numpy as np
from keras.optimizers import SGD
from keras.models import model_from_yaml,model_from_json
from keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(suppress=True,linewidth=300)

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
confusion_matrix=np.zeros((10,10))
accuracy=0
for i,n in enumerate(generator.filenames):
    my_pred=np.argmax(predictions[i])
    column=-1
    if n.startswith("Classical"):
        column = 0
        if my_pred == 0:
            accuracy += 1
    elif n.startswith("Electronic"):
        column = 1
        if my_pred == 1:
            accuracy += 1
    elif n.startswith("Experimental"):
        column = 2
        if my_pred == 2:
            accuracy += 1
    elif n.startswith("Folk"):
        column = 3
        if my_pred == 3:
            accuracy += 1
    elif n.startswith("Hip-Hop"):
        column = 4
        if my_pred == 4:
            accuracy += 1
    elif n.startswith("Instrumental"):
        column = 5
        if my_pred == 5:
            accuracy += 1
    elif n.startswith("International"):
        column = 6
        if my_pred == 6:
            accuracy += 1
    elif n.startswith("Old-Time_Historic"):
        column = 7
        if my_pred == 7:
            accuracy += 1
    elif n.startswith("Pop"):
        column = 8
        if my_pred == 8:
            accuracy += 1
    elif n.startswith("Rock"):
        column = 9
        if my_pred == 9:
            accuracy += 1
    confusion_matrix[my_pred][column]+=1

print(confusion_matrix)
print(accuracy/(np.shape(predictions)[0]))


