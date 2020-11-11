# Unknown Face Recognition

## Abstract

This paper presents a method of recognising and tracking previously unknown faces in real-time video using the 720-pixel MacBook Pro Facetime HD camera. To identify faces in an image a Histogram of Oriented Gradients (HOG) and Support Vector Machine (SVM) based method was used. Embeddings for each face were then generated using a modified ResNet network. These embeddings were then compared using the Euclidean distance and if the distance was below a set threshold of 0.6, then the two embeddings were considered a match. If no match was found, then the embedding was recorded as a new face and used to recognise that face in later images. This method was tested on the Yale Faces dataset and had an accuracy of 98.18 %. In order to improve the runtime of the method, the image was reduced in size. The optimal image reduction was 60 % of the original size as this had no significant effect on the accuracy of the method but was able to achieve 7.7 fps with no display.

## Face Recognition Setup

Instructions for installing the Face Recognition library can be found [here](https://github.com/ageitgey/face_recognition/blob/master/README.md).

## Running the realtime detector

From the `face_rec` directory, execute:

```
python recognition.py
```

Optionally, the following parameters can be used to adjust the results:

* -s (int) Percentage of original size to process the image at (default at 50)
* -d (int) Percentage of original size to display the image at (default at 50)
* -f (directory) directory where pre-labelled images of faces are (default at "faces")

## Running the single image detector

From the `face_rec` directory execute:

```
python recognition.py -i [path to image]
```

An example image has been provided, simply use "test.jpg" as the path:

```
python recognition.py -i test.jpg
```

Optionally, the following parameters can be used to adjust the results:

* -s (int) Percentage of original size to process and display the image at (default at 50)
* -f (directory) directory where pre-labelled images of faces are (default at "faces")

## Evaluate the accuracy using Yale Faces dataset

From the `face_rec` directory execute:

```
python recognition.py -t [path to test directory]
```

Optionally, the following parameters can be used to adjust the results:

* -r (string) Image type fromn Yale Faces dataset to be used as the reference (default at _normal)

Options include: 
* _normal
* _sad
* _happy
* _sleepy
* _surprised
* _wink
* _glasses
* _noglasses
* _rightlight
* _leftlight
* _centerlight
