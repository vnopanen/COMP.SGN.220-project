#  COMP.SGN.220 Project work

> The project aims to classify different musical instruments using CNNs. 

##  General Information

- [IRMAS](https://www.upf.edu/web/mtg/irmas) dataset was used to train, validate and test the system.
- development was done in Python with PyTorch framework

##  Usage

- extract IRMAS-TrainingData and IRMAS-TestingData-Part1 to the current workspace
- run [serialize_data.py](https://github.com/vnopanen/COMP.SGN.220-project/blob/master/serialize_data.py "serialize_data.py") to preprocess the dataset
- run [training.py](https://github.com/vnopanen/COMP.SGN.220-project/blob/master/training.py "training.py") which contains the training, validation and testing procedures

##  Project Status

Project is:  _complete_

##  Room for Improvement

Implement multi classifier

##  Acknowledgements

This project was based on this [paper](http://cs230.stanford.edu/projects_winter_2021/reports/70770755.pdf)
(Detecting and Classifying Musical Instruments with Convolutional Neural Networks) from D. Hing and C. Settle. 
First goal was to convert the work from TensorFlow to PyTorch and then further optimize the project.
