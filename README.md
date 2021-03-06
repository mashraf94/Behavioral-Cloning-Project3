# Behavioral-Cloning-Project3
In this project, deep neural networks and convolutional neural networks are used to clone a driver's behavior. The model is an imitation of the NVIDIA architecture with a couple of enhancements. The proposed model will predict a steering angle to an autonomous vehicle that drives around the tracks.

To autonomously drive the car in Udacity's simulator using the proposed [model](model.h5):

1. Enter `python drive.py model.h5` in terminal
2. Launch Simulator in Autonomous mode

*Please check the [writeup report](writeup_report.md) for further details*

* *[model.py](model.py) includes the python code used for augmenting, processing and filtering the dataset. Plus, the model architecture and training. To train this model, change `DIR_NAMES` to your directories in model.py and run:*
`python model.py`

* *For more indepth on my code, check the IPython Notebook [model.ipynb](model.ipynb)*
