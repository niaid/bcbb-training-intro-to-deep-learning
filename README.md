# Intro to Deep Learning [September 2022]

Example scripts for running deep neural networks using Keras and TensorFlow v2 in Python and R.

## Installing the environment on your own machine

### Python Users

I highly recommend installing packages using [Anaconda or Miniconda](https://repo.anaconda.com/) on your machine. If you do not have it already, you may need IT support to get this install if you do not have admin rights on your machine. Note - some users appear to be having an issue where some conda repositories are being blocked on the NIH network. If you run into this, please try logging out of the VPN and installing again. If you are not on a laptop, you may need IT support to get around this or to download the packages and install them locally.


To run these you'll need Python (version 3.x) and the following major packages and their dependencies installed:
  * tensorflow (version 2 now includes keras)
  * scikit-learn
  * Pillow
  * scipy
  * matplotlib
  * keras-nlp [only for intermediate course]
 
I recommend installing the specific package versions listed in the `deep_learning_environment.yml` file by creating a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). With conda installed, you can easily make an environment called `tflow` in a terminal window using the command:
`conda env create -f deep_learning_environment.yml`

You can then activate this environment via:
`conda activate tflow`

You can install to a specific directory with a custom name for the environment using:
`conda env create --prefix ./envs -n myname -f deep_learning_environment.yml`

Where, `./envs` is the directory you want to install to and `myname` is the name you want to call the environment. When you are done with your analysis, you can simply deactivate the environment with the command:
`conda deactivate`

*Note For Mac Users! - If you run into problems with the scripts crashing, you might also need to also install the `nomkl` package to prevent a multithreading bug in `numpy`. This is not included in the environment YAML file so first activate the `tflow` environment and install with `conda install nomkl`)*

Feel free to use the editor of your choice, but if you are looking for a free python editor with nice graphical user interface (similar to RStudio), I recommend [Spyder](https://www.spyder-ide.org/). You can install this by first activating the `tflow` environment, and then typing (note the version):
`pip install spyder==5.2`

You can then start it inside the `tflow` environment by typing `spyder` (on a Unix system try `spyder &` to have it run in the background). There are also installers available on the Spyder website if you prefer. Note, that [Quatro](https://quarto.org/docs/tools/rstudio.html) also allows RStudio to seamlessly run Python notebooks as well if you prefer.

## R Users

For R users I highly recommend using RStudio to run your scripts and to install your library packages. 
