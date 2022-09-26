#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
keras_cnn_example.py
By: Dan Veltri (dan.veltri@gmail.com)
Code modified from: http://parneetk.github.io/blog/cnn-cifar10/
Date Created: 12.05.2018
Last Updated: 12.09.2021 - Updated plot code for xticks

Here we're going to use a convolutional neural network (CNN)- specifically an
2D CNN to try and predict if images from the CIFAR10 dataset belong to one of
ten classes/categories. For more details on Keras' CNN implementation see:
https://keras.io/layers/convolutional/

!Mac OSX NOTE: Due to OpenMPI issues you may need to run the lines:
		import os
		os.environ['KMP_DUPLICATE_LIB_OK']='True'
	If still having issues install the Python package 'nomkl' to prevent OpenMPI from crashing

The problems we need to solve to use our CNN are:

1) How do we 'massage' our image data and responses so that it fits into our network?
	- We'll have to do some reshaping first!
	
2) Parameters - How good of performance can you get?
	- Try adjusting the number of epochs, filters and kernal sizes 
	
The CIFAR10 Data: Keras comes with a pre-processed training/testing data set (cifar10) that
includes 50,000 32x32 color (RGB) images labeled as one of ten classes. Load the data as follow:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train and x_test are arrays containing RGB images (num_samples, 3, 32, 32)
y_train and y_test contain arrays of corresponding category numbers (0-9)

More dataset details available in: https://keras.io/datasets/

Challenge:  Can you add additional Conv and pooling layers to the model and improve the ACC?
'''

from __future__ import print_function #python3 printing

# These two lines likely only needed on Mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf # we can now access tf.keras as needed
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model # needed to load saved models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# User set model params
num_filters = 32    # Number of filters to apply to image
kern_shape = (5,5)  # kernel size of filters to slide over image
stride_size = (1,1) # How for to move/slide kernel
num_epochs = 5     # Rounds of training
num_batches = 32    # No. of samples per patch to train at a time


# Function to enable us to plot our training history
def plot_model_history(model_history, save_plot=True, plot_filename='train_history_plot.png'):
    '''
    On entry: model_history is output of keras model.fit, save_plot is boolean, plot_filename is save location
    On exit: If save_plot is True, figure saved to plot_filename path, else matplot figure is shown.    
    '''
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy on left plot
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1,len(model_history.history['accuracy'])/10))
    axs[0].legend(['train', 'validation'], loc='best')
    
    # summarize history for loss on right plot
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1,len(model_history.history['loss'])/10))
    axs[1].legend(['train', 'test'], loc='best')
    
    if save_plot:
        plt.savefig(plot_filename) #if you prefer to save a local copy
        plt.close(fig)
    else:
        plt.show()
    return

# Load in image data and responses
print("Loading in data.")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("\n\nLoaded {} training examples with {} responses and {} testing examples with {} responses.".format(len(x_train),len(y_train),len(x_test),len(y_test)))


# Reshape and normalize the image data. Adjust the responses to be categorical
x_train = x_train.reshape(x_train.shape[0],32,32,3).astype('float32')/255
x_test = x_test.reshape(y_test.shape[0],32,32,3).astype('float32')/255
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Let's just train on the first few thousand samples to speed things up!
# Note we shouldn't expect good performance with a sample this small.
x_train = x_train[0:2000,:]
y_train = y_train[0:2000,:]

# Define our sequential model
print("\nBuilding model...")
model = Sequential()
model.add(Conv2D(num_filters,
                 kernel_size=kern_shape,
                 strides=stride_size,
                 padding='same',
                 input_shape=(32,32,3))) #this needs to fit our image dimensions and no. of color chanels (RGB=3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax')) #we need outputs in one of ten categories
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

print("\nTraining now...")
model_info = model.fit(x_train, y_train, 
                       epochs=num_epochs,
                       batch_size=num_batches,
                       validation_data = (x_test,y_test),
                       verbose=1)

plot_model_history(model_info, save_plot=False)

print("\nMaking predictions...")
result = model.predict(x_test)

#Calc ACC stats
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
acc = float(num_correct)/len(result) * 100.0
print("\nModel Testing Accuracy: {}%".format(np.round(acc,2)))

# END PROGRAM
