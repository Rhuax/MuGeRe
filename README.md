# MuGeRe
MUsic GEnre REcognition

This is an attempt to perform music genre recognition directly from audio with deep convolutional neural networks.
The basic process goes from generating spectrograms using "SoX" into png files. PNG files are chunked and each chunk represents a training instance for the learning algorithm.
Every chunk is then fed to the CNN which extracts salient features to be passed to fully connected layers in order to perform categorical prediction.
![alt tag](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000424_0.png?token=AIqauXQAeYm2ZiBAs0fMjCfeifgCDsVHks5aA1VPwA%3D%3D)
![alt tag](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000615_0.png?token=AIqaufiQL6qoe1NNP3ri6Fnl7jT_rUhIks5aA1VswA%3D%3D)
![alt tag](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000897_3.png?token=AIqauW2mN3x0kIHee-PQ1KBQTcJZYiXaks5aA1WMwA%3D%3D)
![alt tag](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/006674_3.png?token=AIqauVPYVQhEi_OXxpOTmXnQx-n-SJ0Bks5aA1WhwA%3D%3D)
![alt tag](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/024512_4.png?token=AIqauV4k-GEnzCA4GhApXSYiJ8z9yJfzks5aA1WvwA%3D%3D)

# Requirements

* Tensorflow
* Keras
* Pillow
* Pydub
* sox
* ffmpeg