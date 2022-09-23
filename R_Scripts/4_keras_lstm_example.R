#!/usr/bin/env Rscript
#
# keras_lstm_example.R
# By: Dan Veltri (dan.veltri@gmail.com)
# Date: 02.05.2018
# Updated: 09.22.2022 - Updates for TF >2.6 and pROC syntax
# Code modified from : https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# R-code adjustments from: https://keras.rstudio.com
#
# Here we're going to use a recurrent neural network- specifically an
# LSTM (Long Short-Term Memory) to try and predict positive or negative
# movie reviews in IMDB. For more details on Keras' LSTM implementation see:
# https://keras.io/layers/recurrent/#lstm
#
# The problems we need to solve to use our LSTM are:
# 1) How do we encode our movie reviews in a way that the network can understand?
#	- We'll have to use an embedding layer to map our words (as numbers) into a
#	 vector of  size "embedding_vector_length".
#	
# 2) Reviews are not all the same length- how do we deal with variable size input?
#	- We'll only look at a certain number of "most frequent" review words
#	 (top_words variable).
#	- We can also use the keras.preprocessing library to put a limit on the length
#	 of our reviews (cut them off at max_review_length). We can "pad" any shorter
#	 reviews with 0's that the network learns to ignore.
#	
# 3) Parameters - How good of performance can you get?
#	- Try adjusting the number of top words, review length, the size of our embedding
#	 vector, the number of LSTM units.
#	
# The IMDB Data: Keras comes with a pre-processed training/testing data set (imbd) that
# includes 25,000 movie reviews labeled by "positive" or "negative" reviews. The dataset
# has already encoded the words as numbers (word indexes arranged by frequency of the top
# 10,000 words). Accordingly word "3" is the third-most frequent word in the corpus. The
# number 0 has been reserved as a padding character. Load the data as follow:
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
# x_train and x_test contain the lists of integers (word indexes)
# y_train and y_test contain corresponding respons (0=bad review, 1=good review)
#
# More dataset details available in: https://keras.io/datasets/
#
# Challenge:  Is it right to really tune our parameters using the test set? Can you modify
# the code to make a validation set? Can you make the LSTM Bidirectional?
#=============================================================================================================

library(keras)
library(pROC) #provides auc for auROC

# Define the top words, review size, and model params
top_words <- 500
max_review_length <- 200
embedding_vector_length <- 32
num_lstm_units <- 25
num_epochs <- 10
num_batches <- 32

# Load in data and pad reviews shorter than 'max_review_length' with 0's in front
print("Loading in data.")

imdb <- dataset_imdb(num_words = top_words)

x_train <- imdb$train$x %>% pad_sequences(maxlen = max_review_length) 
x_test <- imdb$test$x %>% pad_sequences(maxlen = max_review_length)

# To speed things up on a laptop lets only use a few thousand examples
x_train <- x_train[1:2000,]
y_train <- imdb$train$y[1:2000]
x_test <- x_test[1:2000,]
y_test <- imdb$test$y[1:2000]

print(paste0("Loaded ", nrow(x_train), " training examples with ", length(y_train), " responses and ", nrow(x_test)," testing examples with ", length(y_test)," responses."))

# Initialize sequential model. Note the pipe operator %>% is used to chain model layers
model <- keras_model_sequential()
model %>%
  layer_embedding(top_words, embedding_vector_length, input_length = max_review_length) %>%
  layer_lstm(units = num_lstm_units, activation = 'tanh', return_sequences = FALSE, stateful = FALSE) %>%
  layer_dense(1, activation="sigmoid")

# Compile model
model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "accuracy")

summary(model)

print("Training now...")

# Let's evaluate just the first 1000 entries so that this doesn't take so long!
history <- model %>% fit(x_train, y_train, batch_size = num_batches, epochs = num_epochs, validation_data = list(x_test, y_test))

# Plot out training history
plot(history)

print("Testing prediction performance...")
preds <- model %>% predict(x_test) %>% `>`(0.5) %>% k_cast(dtype="int32") # if this wasn't binary prediction, after predict pipe to k_argmax

# Calculate performance stats
cm <- data.frame(tn=0,fn=0,fp=0,tp=0) #confusion matrix
cm[, c("tn","fn","fp","tp")] <- table(Actual=y_test,Predicted=as.list(preds))
sens <- cm$tp / (cm$tp + cm$fn) * 100.0
spec <- cm$tn / (cm$tn + cm$fp) * 100.0
prec <- cm$tp / (cm$tp + cm$fp) * 100.0
acc <- (cm$tp + cm$tn) / (cm$tn + cm$fp + cm$fn + cm$tp) * 100.0
auroc <- roc(y_test, as.list(preds), levels=c(0,1), direction='<') %>% auc() #from pROC package

# For MCC here we use as.numerics to prevent overflows, R-ints are still 32bit!
mcc_numerator <- as.numeric(cm$tp * cm$tn) - as.numeric(cm$fp * cm$fn) 
mcc_denominator <- sqrt(as.numeric(cm$tp + cm$fp) * as.numeric(cm$tp + cm$fn) * as.numeric(cm$tn + cm$fp) * as.numeric(cm$tn + cm$fn)) 

cat("TP: ", cm$tp, ", TN: ", cm$tn, ", FP: ", cm$fp, ", FN: ", cm$fn)
cat(" Sensitivity: ", round(sens,4))
cat(" Specificity: ", round(spec,4))
cat(" Accuracy: ", round(acc,4))
cat(" MCC: ", round(mcc_numerator/mcc_denominator,4))
cat(" Area Under ROC: ", round(auroc,4))
cat(" Precision: ", round(prec,4))

# END OF PROGRAM
