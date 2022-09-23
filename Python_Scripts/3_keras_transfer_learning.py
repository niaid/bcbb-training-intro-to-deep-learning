#!/usr/bin/env python
# -*- coding: utf-8 -*-
''''
keras_transfer_learning_malaria.py
By: Dan Veltri (dan.veltri@gmail.com)
Date Created: 12.09.2018
Last Updated: 12.09.2021 - Updated plot code for xticks

Original Paper Concept from: S. Rajaraman et al. PeerJ 6:e4568, 2018.
Some code modified from: https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/
Plot history function modified from: http://parneetk.github.io/

Here we're going to perform 'transfer learning', borrowing from a pre-trained model 'MobileNet' and using
is to learn a new classifier for images of malaria parasitized vs uninfected cells.

On a laptop this code is probably best for just learning the basic concept of using pre-trained models and image
generators- you'd really need to run this on a cluster to process images enough to get a decent classifier. 

!NOTE: You will need to set the 'train_img_location' and 'test_img_location' paths to the dataset
!NOTE: You will likely need to install the Python package 'Pillow' to make this run properly with images.
!Mac OSX NOTE: Due to openmpi issues you may need to run the lines:
		import os
		os.environ['KMP_DUPLICATE_LIB_OK']='True'
	If still having issues install the Python package 'nomkl' to prevent OpenMPI from crashing

Malaria Dataset (source: https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip)
The full-size dataset contains about 50K parasitized/uninfected cells for training and 10K for testing.
Images are variable size PNGs (around ~250x250px) in RGB color. 
'''

from __future__ import print_function #python3 printing

# These two lines likely only needed on Mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import PIL # required by keras.utils load_img function

import tensorflow as tf # we can now access tf.keras as needed
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras import Sequential
#from tensorflow.keras.models import load_model # needed to load saved models
#from keras.layers import Dense,GlobalAveragePooling2D
#from tf.keras.applications import MobileNet


# User set model parameters
train_datagen_shear = 0.2    # % to shear trian imgs for more variety
train_datagen_zoom = 0.2     # % to zoom train imgs for more variety
target_img_shape = (128,128) # Dim. of target images
num_epochs = 10              # Rounds of training
num_batches = 32             # No. of samples per patch to train at a time

# Paths to train/test images - make sure these are present and unzipped!
train_img_location = '../cell_images_mini/Train'
test_img_location = '../cell_images_mini/Test'


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


# Import our dataset using generators- these make it easy for Keras to process many input files
# Images should be in Train and Test folders and within these should be two subfolders,
# one for each class with the actual pics. Class folder names should be consistent!
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=train_datagen_shear,
        zoom_range=train_datagen_zoom,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_img_location,
        target_size=target_img_shape,
        color_mode='rgb',
        batch_size=num_batches,
        class_mode='binary',
        shuffle=True)

test_generator = test_datagen.flow_from_directory(
        test_img_location,
        target_size=target_img_shape,
        color_mode='rgb',
        batch_size=num_batches,
        class_mode='binary',
        shuffle=True)

# Import MobileNet model and discards the top dense layer.
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# We add new layers to the model to help learn our particular problem
# This produces a tensor obj, not a model layer at this point
pred_tensor = base_model.output
pred_tensor = GlobalAveragePooling2D()(pred_tensor)
pred_tensor = Dense(1024,activation='relu')(pred_tensor)
pred_tensor = Dense(1024,activation='relu')(pred_tensor)
pred_tensor = Dense(512,activation='relu')(pred_tensor) 
pred_tensor = Dense(1,activation='sigmoid')(pred_tensor) 

model=Model(inputs=base_model.input, outputs=pred_tensor)

# Freeze some pretrained layers so we don't overwrite everything! 
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

print("\nTraining now...")

# Note when using generators like this we specify 'steps_per_epoch' rather than 'batch_size'
model_info = model.fit( train_generator,
						epochs=num_epochs,
						steps_per_epoch=(train_generator.n / train_generator.batch_size),
						validation_data=test_generator,
						validation_steps=(test_generator.n / test_generator.batch_size),
						verbose=1)

print("\nMaking predictions on test set...")
results = model.evaluate(	test_generator,
							steps=(test_generator.n / test_generator.batch_size))

print('Final Test ACC:',(results[1]*100.0))

plot_model_history(model_info, save_plot=False)

# END PROGRAM
