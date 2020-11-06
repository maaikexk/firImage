# README

_Written by Maaike Keurhorst & Daan Kooij._

To run the program, first download the W17 dataset from [GitHub](https://github.com/marialeyva/TB_Places), extract it, and place it in the data directory of the project. Afterwards, the ```main.py``` file can be run using your Python 3.8 interpreter (given that you have the necessary dependencies defined, which are defined at the top of the Python source files). For the training of the TCNN network, a GPU is highly recommended, as this can massively speed up the process (still, it can take a few hours on modern hardware). To allow the program to run on a GPU, follow the instructions from the [TensorFlow](https://www.tensorflow.org/install/gpu) website.

The reason that we use normal Python source files instead of a Python notebook, is so that we can neatly divide our functions over the appropriate files, so that the structure of the program becomes more logical and easier to maintain.