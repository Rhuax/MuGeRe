import numpy as np
from keras.optimizers import SGD
from keras.models import model_from_json
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

# Initialization
#confusion_matrix = np.zeros((10, 10))
right = np.zeros(10)
wrong = np.zeros(10)
accuracy = 0
song_genres = np.zeros(10)
spectrogram = -1

predictions = model.predict_generator(generator, steps=482)
#np.savetxt("predictions.txt", predictions)
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
    song_genres += predictions[i]            # Controllare

print("Correctly classified songs: ")
print(right)
print("Misclassified songs: ")
print(wrong)
print("Total accuracy on test set: ")
print(accuracy/((np.shape(predictions)[0])//5))


