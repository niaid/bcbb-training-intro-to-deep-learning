#!/usr/bin/env Rscript
#
# keras_hello_world_functional_api.R
# By: Dan Veltri (dan.veltri@gmail.com)
# Date: 9.04.2022
#
# Extending our 'hello world' example to use the Functional API
# instead of the Sequential API.
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
num_epochs <- 100  # Rounds of training
num_batches <- 16 # No. of samples per patch to train at a time


# Randomly split data into even train/test (set too small for validation)
tr_idx <- sample(nrow(iris),75)
te_idx <- nrow(iris) - tr_idx
x_train <- as.matrix(iris[tr_idx,1:4])
y_train <- to_categorical(as.numeric(iris[tr_idx,5])-1,3)
x_test <- as.matrix(iris[te_idx,1:4])
y_test <- to_categorical(as.numeric(iris[te_idx,5])-1,3)


# Build the structure of our model now using the Functional API
# This requires 3 majors things:
# - An input layer(s) 
# - An output series of layers
# - A 'keras_model' function call where we plug in our inputs and outputs.
#   if using multiple input/output layers pass them as a list
# We often define the "work" layers in a separate variable (say 'x') so that
# we can easily reuse our inputs multiple times if we want later
# Note the Functional API can get more complicated than the Sequential API
# so for this reason we're going to provide a name to each component so we
# can keep better track of things - this makes the model summary more
# readible and useful for debugging when needed!

iris_inputs <- layer_input(shape=c(4), name="iris_input_feats")

x <- iris_inputs %>%
  layer_dense(units=200, name="dense_layer1") %>%
  layer_dense(units=100, name="dense_layer2")

iris_class_outputs <- x %>%
  layer_dense(units=3, activation="softmax", name="final_softmax_output")

model <- keras_model(inputs=iris_inputs, outputs=iris_class_outputs, name='iris_model')

# From here we can compile and fit like we did before!
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