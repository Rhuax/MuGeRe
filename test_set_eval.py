import numpy as np
from keras.optimizers import SGD
from keras.models import model_from_yaml,model_from_json
from keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(suppress=True,linewidth=300)

img_width = 160
img_height = 150


def compare_song_class(prediction, genre):
    if genre.startswith("Classical") and prediction == 0:
        return prediction
    elif genre.startswith("Electronic") and prediction == 1:
        return prediction
    elif genre.startswith("Experimental") and prediction == 2:
        return prediction
    elif genre.startswith("Folk") and prediction == 3:
        return prediction
    elif genre.startswith("Hip-Hop") and prediction == 4:
        return prediction
    elif genre.startswith("Instrumental") and prediction == 5:
        return prediction
    elif genre.startswith("International") and prediction == 6:
        return prediction
    elif genre.startswith("Old-Time_Historic") and prediction == 7:
        return prediction
    elif genre.startswith("Pop") and prediction == 8:
        return prediction
    elif genre.startswith("Rock") and prediction == 9:
        return prediction
    else:
        return -1

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
confusion_matrix = np.zeros((10, 10))
accuracy = 0
song_genres = np.zeros(10)
spectrogram = -1

predictions = model.predict_generator(generator, steps=482)
#np.savetxt("predictions.txt", predictions)
for i, n in enumerate(sorted(generator.filenames)):
    my_pred = np.argmax(predictions[i])
    spectrogram += 1
    if spectrogram == 5:
        # If prediction is valid, update accuracy
        column = compare_song_class(np.argmax(song_genres), n)      # Controllare
        if column != -1:
            accuracy += 1
        # Reset all variables
        spectrogram = -1
        song_genres = np.zeros(10)
        # Update confusion matrix
        # La matrice di confusione è andata a puttane perché quando non azzecca perdo il valore di column che è a -1, kek sry
        confusion_matrix[my_pred][column] += 1
    else:
        song_genres[my_pred] += predictions[i][my_pred]             # Controllare

print(confusion_matrix)
print(accuracy/((np.shape(predictions)[0])//5))


