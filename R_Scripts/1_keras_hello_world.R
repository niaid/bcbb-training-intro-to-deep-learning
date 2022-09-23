#!/usr/bin/env Rscript
#
# keras_hello_world.R
# By: Dan Veltri (dan.veltri@gmail.com)
# Date: 12.04.2018
#
# A basic 'hello world' example to test your Keras+TensorFlow install.
#
# Throughout this course - we'll save features as 'X', and classes
# as 'Y'. So x_train are features of training samples and y_train are
# matching 'ground truth labels for the species of Iris the sample is.
#
# Iris Data Set: Iris data set is built into R
# Comprised of 3 different Iris species (50 samples each as rows)
# Each flower has 4 features, width and height of sepals and petals
# Features are in the first for cols, flower "class" label is final col
# Challenge: Prediction performance is not so great - can you modify
# the model to get better performance? 
#
#======================================================================

library(keras)

data(iris)
summary(iris)

# These are some basic settings for how to train our model
num_epochs <- 10  # Rounds of training
num_batches <- 16 # No. of samples per patch to train at a time


# Randomly split data into even train/test (set too small for validation)
tr_idx <- sample(nrow(iris),75)
te_idx <- nrow(iris) - tr_idx
x_train <- as.matrix(iris[tr_idx,1:4])
y_train <- to_categorical(as.numeric(iris[tr_idx,5])-1,3)
x_test <- as.matrix(iris[te_idx,1:4])
y_test <- to_categorical(as.numeric(iris[te_idx,5])-1,3)

# Build the structure of our model (no training yet!)
# We're going to establish a sequential model
# Note the pipe operator %>% is used to chain model layers
# The final 'softmax' activation squishes predictions between 0-1.
# Q: Why is the number of units in the last layer set to 3?
model <- keras_model_sequential() %>%
  layer_dense(units=500, input_shape=c(4)) %>%
  layer_dense(units=3, activation="softmax")

# Compile our model- need to do this before using it
model %>% compile(optimizer="sgd",
                  loss="categorical_crossentropy",
                  metrics="accuracy")

summary(model)

# Do the actual training!
print("Training now...")
model_info <- model %>% fit(x_train, y_train,
                            batch_size=num_batches,
                            epochs=num_epochs,
                            validation_data=list(x_test,y_test))

print(model_info)

# Example of saving and reloading a model
save_model_hdf5(model, "my_model.h5", overwrite = TRUE)
rm(model)
model <- load_model_hdf5("my_model.h5")

# Let's see how we did!
print("Making predictions...")
scores <- model %>% evaluate(x_test, y_test)
cat("Testing Loss: ", scores['loss'], "| Testing Accuracy: ", scores['accuracy'] * 100.0, "%") 

# End Program