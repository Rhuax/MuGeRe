import numpy as np
from keras.optimizers import SGD, rmsprop
from keras.models import model_from_json, model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(suppress=True,linewidth=300)

img_width = 160
img_height = 150


def compare_song_class(prediction, genre):
    if genre.startswith("Classical") and prediction == 0:
        return True
    elif genre.startswith("Electronic") and prediction == 1:
        return True
    elif genre.startswith("Experimental") and prediction == 2:
        return True
    elif genre.startswith("Folk") and prediction == 3:
        return True
    elif genre.startswith("Hip-Hop") and prediction == 4:
        return True
    elif genre.startswith("Instrumental") and prediction == 5:
        return True
    elif genre.startswith("International") and prediction == 6:
        return True
    elif genre.startswith("Old-Time_Historic") and prediction == 7:
        return True
    elif genre.startswith("Pop") and prediction == 8:
        return True
    elif genre.startswith("Rock") and prediction == 9:
        return True
    else:
        return False


def def_genre_from_str(genre):
    if genre.startswith("Classical"):
        return 0
    elif genre.startswith("Electronic"):
        return 1
    elif genre.startswith("Experimental"):
        return 2
    elif genre.startswith("Folk"):
        return 3
    elif genre.startswith("Hip-Hop"):
        return 4
    elif genre.startswith("Instrumental"):
        return 5
    elif genre.startswith("International"):
        return 6
    elif genre.startswith("Old-Time_Historic"):
        return 7
    elif genre.startswith("Pop"):
        return 8
    elif genre.startswith("Rock"):
        return 9

f = open('arch_weights/MuGeRe Architecture_F.yaml', 'r')
model = model_from_yaml(f.read())
f.close()
model.load_weights('arch_weights/MuGeRe Weights_F.h5')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

# Initialization
confusion_matrix = np.zeros((10, 10))
right = np.zeros(10)
wrong = np.zeros(10)
accuracy = 0
song_genres = np.zeros(10)
spectrogram = -1

predictions = model.predict_generator(generator, steps=482)
for i, n in enumerate(sorted(generator.filenames)):
    my_pred = np.argmax(predictions[i])
    spectrogram += 1
    if spectrogram == 4:
        # If prediction is valid, update accuracy
        if compare_song_class(np.argmax(song_genres), n):
            accuracy += 1
            right[my_pred] += 1
        else:
            wrong[my_pred] += 1
        # Reset all variables
        spectrogram = -1
        song_genres = np.zeros(10)
    confusion_matrix[my_pred][def_genre_from_str(n)] += 1
    song_genres += predictions[i]


for i in range(10):
    if i == 0:
        print("Classica")
    elif i == 1:
        print("Electronic")
    elif i == 2:
        print("Experimental")
    elif i == 3:
        print("Folk")
    elif i == 4:
        print("Hip-Hop")
    elif i == 5:
        print("Instrumental")
    elif i == 6:
        print("International")
    elif i == 7:
        print("Old Time Historic")
    elif i == 8:
        print("Pop")
    elif i == 9:
        print("Rock")
    print('{} su {} - Percentuale {}'.format(right[i], right[i] + wrong[i], right[i] / (right[i] + wrong[i])))
print("Total accuracy on test set: ")
print(accuracy/((np.shape(predictions)[0])//5))
print("\nConfusion matrix:")
print(confusion_matrix)


