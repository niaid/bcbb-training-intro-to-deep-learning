# Intro to Deep Learning [September 23, 2022]

Example scripts for running deep neural networks using Keras and TensorFlow2 in Python and R.

## Installing the environment on your own machine

To run these you'll need python and the following packages installed. :
  * numpy 
  * scikit-learn
  * h5py
  * Pillow
  * matplotlib
  * tensorflow (version 2 now includes keras)
  
I recommend installing packages using a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/). On a Linux machine, `pip` should work for the above packages but if you have Anaconda installed, you can easily use the `deep_learning_environment.yml` file to make a `deep_learning` environment via the command:
`conda env create -f deep_learning_environment.yml`.

You can install to a specific directory with a custom name for the environment using: `conda env create --prefix ./envs -n myname -f deep_learning_environment.yml`  where `./envs` is the directory you want to install to and `myname` is the name you want to call the environment. 

*Note For Mac Users! - If you run into problems with the scripts crashing, you might also need to also install the `nomkl` package to prevent a multithreading bug in `numpy`.*
