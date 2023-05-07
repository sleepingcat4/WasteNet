Repository contains code to implement Uk government’s leading Waste classification Algorithm. The code for the published paper was not available neither any explicit instructions except for the methodology and novel approach that they took. 

## Introduction


WasteNet shows superior capability in classifying images over previously known pre-trained models such as ResNet-50, AlexNet and DenseNet. It uses a novel technique proposed by the authors of the paper i.e. Hybrid Tuning that leverages both feature extraction and fine-tuning. 

A approach known as discriminative learning was used to prevent “catastrophic forgetting” and overall generalise the dataset and convergence. 

Dataset

WasteNet was originally trained on trashnet dataset which was open sourced by Stanford University. It contains 2533 images of waste in 6 different categories. 

## Data Augmentation


Originally several data augmentation techniques were used on the dataset before feeding the final version of the dataset. But, in our research and implementation of the paper, we focused more on the Model’s Architecture and achieving a good overall score. 

## File-structure
Augmentation contains a python function to augment your dataset as per mentioned in the original paper. model.py & train.py respectively contain the model architecture and training code for training it on CIFAR-10 dataset. 

## Training

In our training, we assumed the hyperparameters since it was not explicitly mentioned in the paper, and trained the model for 20 epochs. Though the original paper trains the model for 1000 epochs to achieve convergence and 97% accuracy. Highest known till date. 

## Development

The Model and paper is still under-development. Though we have already implemented the original model architecture as mentioned in the WasteNet paper, we are thinking of changing a few lines and features of our own and adding some. We’ll continue to update the code. 

## Licence

This repository falls under a modified MIT Licence, which means you can’t sublicense/sell the software/algorithm/artificial intelligence written in this repository without the explicit permission from **sleepingcat**. And the code in this repository can only be used for research purposes strictly. 

**Anything else will be a clear violation of Copyright Law.**
