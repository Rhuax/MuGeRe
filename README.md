# MuGeRe
MUsic GEnre REcognition

![alt text](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000424_0.png)![alt text](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000897_3.png)![alt text](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/024512_4.png)![alt text](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/006674_3.png)![alt text](https://raw.githubusercontent.com/Rhuax/MuGeRe/master/readme_images/000615_0.png)



This is an attempt to perform music genre recognition directly from audio with deep convolutional neural networks.
The basic process goes from generating spectrograms using "SoX" into png files. PNG files are chunked and each chunk represents a training instance for the learning algorithm.
Every chunk is then fed to the CNN which extracts salient features to be passed to fully connected layers in order to perform categorical prediction. 


# Requirements

* Tensorflow
* Keras
* Pillow
* Pydub
* sox
* ffmpeg
