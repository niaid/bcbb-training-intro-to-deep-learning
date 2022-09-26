#!/usr/bin/env Rscript
#
# keras_transfer_learning_malaria.R
# By: Dan Veltri (dan.veltri@gmail.com)
# Date: 12.09.2018
# Updated: 09.22.2022 - Updates for TF >2.6 syntax 
#
# Original Paper Concept from: S. Rajaraman et al. PeerJ 6:e4568, 2018.
# Some code modified from: https://www.pyimagesearch.com/2018/12/03/deep-learning-and-medical-image-analysis-with-keras/
# R-code adjustments from: https://keras.rstudio.com
#
# Here we're going to perform 'transfer learning', borrowing from a pre-trained model 'MobileNet' and using
# is to learn a new classifier for images of malaria parasitized vs uninfected cells.
#
# On a laptop this code is probably best for just learning the basic concept of using pre-trained models and image
# generators- you'd really need to run this on a cluster to process images enough to get a decent classifier. 
#
# !NOTE: You will need to set the 'train_img_location' and 'test_img_location' paths to the dataset
# !NOTE: You will likely need to install the Python package 'Pillow' to make this run properly with images.
# !Mac OSX NOTE: I needed to install the Python package 'nomkl' to prevent OpenMPI from crashing
#
# Malaria Dataset (source: https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip)
# The full-size dataset contains about 50K parasitized/uninfected cells for training and 10K for testing.
# Images are variable size PNGs (around ~250x250px) in RGB color. 
#=============================================================================================================

library(keras)

# User set model parameters
train_datagen_shear <- 0.2    # % to shear trian imgs for more variety
train_datagen_zoom <- 0.2     # % to zoom train imgs for more variety
target_img_shape <- c(128,128)  # Dim. of target images
num_epochs <- 5               # Rounds of training
num_batches <- 32             # No. of samples per patch to train at a time

# Paths to train/test images
train_img_location <- '../cell_images_mini/Train'
test_img_location <- '../cell_images_mini/Test'

# Import our dataset using generators- these make it easy for Keras to process many input files
# Images should be in Train and Test folders and within these should  be two subfolders,
# one for each class with the actual pics. Class folder names should be consistent.
train_datagen <- image_data_generator(
  rescale = 1/255,
  shear_range = train_datagen_shear,
  zoom_range = train_datagen_zoom,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(
  rescale = 1/255
)

train_generator <- flow_images_from_directory(train_img_location,
  generator = train_datagen,
  target_size = target_img_shape,
  color_mode = 'rgb',
  class_mode='binary',
  batch_size = num_batches,
  shuffle = TRUE
)

test_generator <- flow_images_from_directory(test_img_location,
  generator = test_datagen,
  target_size = target_img_shape,
  color_mode = 'rgb',
  class_mode='binary',
  batch_size = num_batches,
  shuffle = TRUE
)

# Import MobileNet model and discards the top dense layer.
base_model <- application_mobilenet(weights='imagenet',include_top=FALSE) 

# We add new layers to the model to help learn our particular problem
pred_layers <- base_model$output %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(1024,activation='relu') %>%
  layer_dense(1024,activation='relu') %>%
  layer_dense(512,activation='relu') %>%
  layer_dense(1,activation='sigmoid')

expanded_model <- keras_model(inputs=base_model$input, outputs=pred_layers)

# Freeze some pretrained layers so we don't overwrite everything! 
freeze_weights(expanded_model, from=1, to=20)
unfreeze_weights(expanded_model, from=21) # make layer 21+ trainable

expanded_model %>% compile(optimizer='adam', loss='binary_crossentropy', metrics = "accuracy")

print("Training now...")
train_history <- expanded_model %>% fit_generator(
  generator = train_generator,
  steps_per_epoch = (train_generator$n / train_generator$batch_size),
  epochs = num_epochs,
  verbose = 1)

plot(train_history)

print("Testing prediction performance...")
scores <- expanded_model %>% evaluate_generator(
  generator=test_generator,
  steps=(test_generator$n / test_generator$batch_size),
)

cat("Testing Loss: ", scores['loss'], "| Testing Accuracy: ", scores['accuracy'] * 100.0, "%")

# END PROGRAM
