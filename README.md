Manuel Macedonio ID: 91998858
# Keras Convolutional Neural Network
 Python program used to train a convolution neural network using the Keras API

This project creates a Convolutional Neural Network
and trains it recognize 28 x 28 pixel images.

The project uses Keras API and TensorFlow backend
to create a model and use the training and testing sets provided by Keras.datasets.

Files:
  training.py: Creates a Keras model with layers and
                activation functions. Trains the model
                and saves it to output model.h5.
  model.h5: Trained Model
  testing.py: Loads the trained model and tests it.   
              Outputs the accuracy.
  training_output.PNG: Screenshot of training.py
  testing_output.PNG: Screenshot of testing.py

Initialization:
  Ensure that both Keras and TensorFlow libraries are
  installed to run the files.

Usage:
  training.py:
      [>>] python training.py
  testing.py:
      [>>] python testing.py [model file]
              Ex: python testing.py model.h5

Project can be found on GitHub:
  https://github.com/mmacedon/Keras-Convolutional-Neural-Network
