## R-version of Keras Scripts

For running the Keras R examples you'll need the following libraries:
  * tensorflow
  * reticulate
  * keras
  * pROC
  
After installing the `tensorlow` pacakge, you need to start R and load the library before running the following function: `install_tensorflow()`

This will use the `reticulate` package (which interfaces R with Python code) to install `tensorflow` in the background for use. Without this installation the code will not work! Note if you update the `tensorflow` package in the future, you may need to rerun the `install_tensorflow()` function as well.


