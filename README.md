# A Convolutional Attention Network for Extreme Summarization of Source Code

Implementation of [A Convolutional Attention Network for Extreme Summarization of Source Code](https://arxiv.org/abs/1602.03001) in PyTorch using TorchText

Using Python 3.6, PyTorch 0.4 and TorchText 0.2.3.

**Note**: only the *Convolutional Attention Model* currently works, the *Copy Convolutional Attention Model* is in progress.

To use:

1. `download.sh` to grab the dataset
1. `python preprocess.py` to preprocess the dataset into json format
1. `python run_conv.py` to run the Convolutional Attention Model with default parameters

Use `python run_conv.py -h` to see all the parameters that can be changed, e.g. to run the model on a different Java project within the dataset, use: `python run_conv.py --project {project name}`.