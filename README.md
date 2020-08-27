# Stanford Cars Classification


## Overview
Image classification (196 types of cars) by using deep residual network.

All resources are downloaded from http://ai.stanford.edu/~jkrause/cars/car_dataset.html.


## File description
+ `cars_meta.mat` - Contains a cell array of class names, one for each class.

+ `cars_train_annos.mat` - Contains the variable 'annotations', which is a struct array of length num_images and where each element has the fields:
  + bbox_x1: Min x-value of the bounding box, in pixels.
  + bbox_x2: Max x-value of the bounding box, in pixels.
  + bbox_y1: Min y-value of the bounding box, in pixels.
  + bbox_y2: Max y-value of the bounding box, in pixels.
  + class: Integral id of the class the image belongs to.
  + fname: Filename of the image within the folder of images.

+ `cars_test_annos.mat` - Same format as `cars_train_annos.mat`.

+ `cars_test_annos_without_labels.mat` - Same format as `cars_train_annos.mat`, except the class is not provided.

+ `image_processing.py` - Crop all the image in both training dataset and test dataset.


## Environment
Main package|Version
---|---
Python|3.7.1
Matplotlib|3.1.0
Numpy|1.15.4
PyTorch|1.0.1
