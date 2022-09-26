#!/usr/bin/env Rscript
#
# keras_cnn_example_functional_api_exercise.R
# By: Dan Veltri (dan.veltri@gmail.com)
# Date: 09.20.2022
# Code modified from: http://parneetk.github.io/blog/cnn-cifar10/
# R-code adjustments from: https://keras.rstudio.com
#
# Here we're going to modify our previous Sequential CNN model to use the Functional API
# in Keras to predict if images from the CIFAR10 dataset belong to one of
# ten classes/categories. For more details on Keras' CNN implementation see:
# https://keras.io/layers/convolutional/
# The problems we need to solve to use our CNN are:
# 1) How do we 'massage' our image data and responses so that it fits into our network?
# 	- We'll have to do some reshaping first!
#	
# 2) Parameters - How good of performance can you get?
# 	- Try adjusting the number of epochs, filters and kernal sizes 
#	
# The CIFAR10 Data: Keras comes with a pre-processed training/testing data set (cifar10) that
# includes 50,000 32x32 color (RGB) images labeled as one of ten classes. Load the data as follow:
#  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#  x_train and x_test are arrays containing RGB images (num_samples, 3, 32, 32)
#  y_train and y_test contain arrays of corresponding category numbers (0-9)
#
# More dataset details available in: https://keras.io/datasets/
#
# Challenge:  Can you create two different inputs with Conv networks, one with pooling and one without
# and combine them to make a final output prediction using the Functional API?
#=============================================================================================================

library(keras)

# Define the top words, review size, and model params
num_filters <- 32       # Number of filters to apply to image
kern_shape <- c(5,5)    # kernel size of filters to slide over image
stride_size <- c(1,1)   # How far to move/slide kernel each time
pool_shape <- c(2,2)    # Dim. of max pooling
num_epochs <- 5         # Rounds of training
num_batches <- 32       # No. of samples per patch to train at a time


# Load in data and pad reviews shorter than 'max_review_length' with 0's in front
print("Loading in data.")

cf10 <- dataset_cifar10()

#Reshape and normalize the image data. Adjust the responses to be categorical
x_train <- cf10$train$x/255
x_test <- cf10$test$x/255
y_train <- to_categorical(cf10$train$y, num_classes = 10)
y_test <- to_categorical(cf10$test$y, num_classes = 10)

print(paste0("Loaded ", nrow(x_train), " training examples with ", length(y_train), " responses and ", nrow(x_test)," testing examples with ", length(y_test)," responses."))

######## EXERCISE ########
# Fill in the missing pieces to make this network using the Functional API. Remember to name your layers!
cifar_input <- ### setup input - images have shape 32x32x3

x <- cifar_input %>% ### pass in your inputs and do your actual work here   
  
cifar_output <- x %>%  ### set final network output (10 classes)
  
model <- ### define your model with inputs and outputs

  
# Compile model and fit model per usual
model %>% compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
summary(model)
print("Training now...")
train_history <- model %>% fit(x_train, y_train, batch_size = num_batches, epochs = num_epochs, shuffle=TRUE, validation_data = list(x_test, y_test))

#Plot out training history
plot(train_history)

print("Testing prediction performance...")
scores <- model %>% evaluate(x_test, y_test)
cat("Testing Loss: ", scores['loss'], "| Testing Accuracy: ", scores['accuracy'] * 100.0, "%") 
                              
#END OF PROGRAM
