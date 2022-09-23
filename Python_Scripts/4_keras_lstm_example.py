#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
keras_LSTM_example.py
By: Dan Veltri (dan.veltri@gmail.com)
Date Created: 12.05.2018
Last Updated: 12.09.2021 - Updated plot code for xticks

Code modified from : https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
Plot history function modified from: http://parneetk.github.io/
    
Here we're going to use a recurrent neural network- specifically an
LSTM (Long Short-Term Memory) to try and predict positive or negative
movie reviews in IMDB. For more details on Keras' LSTM implementation see:
https://keras.io/layers/recurrent/#lstm

The problems we need to solve to use our LSTM are:

1) How do we encode our movie reviews in a way that the network can understand?
	- We'll have to use an embedding layer to map our words (as numbers) into a
	 vector of  size "embedding_vector_length".
	
2) Reviews are not all the same length- how do we deal with variable size input?
	- We'll only look at a certain number of "most frequent" review words
	 (top_words variable).
	- We can also use the keras.preprocessing library to put a limit on the length
	 of our reviews (cut them off at max_review_length). We can "pad" any shorter
	 reviews with 0's that the network learns to ignore.
	
3) Parameters - How good of performance can you get?
	- Try adjusting the number of top words, review length, the size of our embedding
	 vector, the number of LSTM units.
	
The IMDB Data: Keras comes with a pre-processed training/testing data set (imbd) that
includes 25,000 movie reviews labeled by "positive" or "negative" reviews. The dataset
has already encoded the words as numbers (word indexes arranged by frequency of the top
10,000 words). Accordingly word "3" is the third-most frequent word in the corpus. The
number 0 has been reserved as a padding character. Load the data as follow:

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

x_train and x_test contain the lists of integers (word indexes)
y_train and y_test contain corresponding respons (0=bad review, 1=good review)

More dataset details available in: https://keras.io/datasets/

Challenge:  Is it right to really tune our parameters using the test set? Can you modify
            the code to make a validation set? Can you make the LSTM Bidirectional?
'''
    
from __future__ import print_function #python3 printing

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt

import tensorflow as tf # we can now access tf.keras as needed
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model # needed to load saved models
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Define the top words, review size, and model params
top_words = 500
max_review_length = 200
embedding_vector_length = 32
num_lstm_units = 25
num_epochs = 10
num_batches = 32

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

# Load in data
print("Loading in data.")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

# Pad reviews shorter than 'max_review_length' with 0's in front
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

#To speed things up on a laptop lets just use a few thousand examples
x_train = x_train[0:2000,]
y_train = y_train[0:2000]
x_test = x_test[0:2000,]
y_test = y_test[0:2000]

print("Loaded {} training examples with {} responses and {} testing examples with {} responses.".format(len(x_train),len(y_train),len(x_test),len(y_test)))

# Define our sequential model
print("\nBuilding model...")
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(units=num_lstm_units, activation='tanh', return_sequences=False, stateful=False)) #sequences need to be returned if you add additional LSTM layers
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

print("\nTraining now...")
model_info = model.fit(x_train, y_train,
                       epochs=num_epochs,
                       batch_size=num_batches,
                       validation_data = (x_test,y_test),
                       verbose=1)

print("\nMaking predictions on test set...")
pred_values = model.predict(x_test)
pred_classes = np.rint(pred_values) #round up or down at 0.5
true_classes = np.array(y_test)

#Get performance stats    
tn, fp, fn, tp = confusion_matrix(y_test,pred_classes).ravel()
auroc = roc_auc_score(true_classes,pred_values) * 100.0
mcc = matthews_corrcoef(true_classes,pred_classes)
acc = (tp + tn) / (tn + fp + fn + tp + 0.0) * 100.0
sens = tp / (tp + fn + 0.0) * 100.0
spec = tn / (tn + fp + 0.0) * 100.0
prec = tp / (tp + fp + 0.0) * 100.0
print("\nModel Testing Binary Classification Performance:\n\nTP: {}\tTN: {}\nFP: {}\tFN: {}\n\nSensitivity: {}%\nSpecificity: {}%\nAccuracy: {}%\nMCC: {}\nArea Under ROC: {}%\nPrecision: {}%".format(tp,tn,fp,fn,np.round(sens,2),np.round(spec,2),np.round(acc,2),np.round(mcc,4),np.round(auroc,2),np.round(prec,2)))

plot_model_history(model_info, save_plot=False)

#END PROGRAM
