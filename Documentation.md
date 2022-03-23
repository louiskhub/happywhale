# Documentation
Implementing artificial Neuronal Networks with Tensorflow course, winter term 2021/22.  
Held by Prof. Dr. Gordon Pipa
Group number 2: Louis Kapp, Felix Hammer, Yannik Ullrich

---
### Table of contents
1. [Introduction](#introduction)
2. [Data Overview](#data_overview)
3. [Data Generation](#data_generation)
4. [Triplet Loss](#triplet_loss)
5. [Model](#model)
6. [Training](#training)
7. [Visualization](#visualization)
8. [Results](#results)
---

<!-- Introduction section -->
<a name='introduction'></a>

## Introduction

This file contains a rough overview of our code for the Project for the Implementing artificial Neuronal Networks with Tensorflow course. The Neuronal Network we propose, aims to differentiate various individuals of whales and should handle new imagedata in a way, that it can map existing individuals to new images and identify new indivudals as well as giving them a new ID.
The data used from [Kaggle](www.kaggle.com) is from the ["Happywhale - Whale and Dolphin Identification"](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data) competition. 
Our code is largely based on [this](https://keras.io/examples/vision/siamese_network/) tutorial.

<!-- Data Overview section -->
<a name='data_overview'></a>

## Data Overview

To work efficiently with a dataset and to explain certain preprocessing and agmentation steps, one first needs to know the datas properties. In ``prelim_overview.ipynb`` we visualized some of the datas properties first. 
In total there are about 51000 training and 28000 test images. For the training images there is a corresponding csv file, which contains a column with the file name of a picture, a column with the name of the species and a column with the ID of a individual. So to create a dataset, one first needs to map the labels (species and ID) to the corrresponding pictures.
In total there are 15000 different individuals from 30 different species. 
In the violin plot we can see, that the height of most images is between 3000 and 4000 pixels and the width between 2000 and 3000 pixels. This is why we decided on a target shape of (300,300) for our processed images.
It is also important to know, that the individuals and corresponding images are not distributed evenly, meaning there are some individuals which have more pictures than others. 

<!-- Data generation section -->
<a name='datageneration'></a>

## Data Generation 
We generate our data in the ``DS_generator``, this includes loading the images as well as directly creating Triplets for our loss function.

### Loading image 

To work with an image in Tensorflow, we first read the file, decode it (in our casse with 3 channels), convert it to tf.float32, resize it to our target shape and expand its dimensions. Then we can use the output to create our dataset.

### Triplet generation

In the case of our Network, we want to work with the Triplet Loss. This means in data generation we already want a Dataset containing Triplets. 
HIER DANN SMART TRIPLET SELECTION ANREIßEN
We know have a Take Dataset containing the anchor, positives as well as negatives. 

<!-- Triplet Loss section -->
<a name='triplet_loss'></a>

## Triplet Loss

We decided to use the Triplet loss in our Neuronal Network. With a Triplet loss, we aim to calculate a Distance between an anchor image and a positive image, as well as an anchor image and a negative image. Because of this we have to create a dataset which contains Triplets of anchor, positive and negative.

<!-- Model section -->
<a name='model'></a>

## Model

To archieve the goal of our Project, we used various components for our Model. 

### Embedding Model

For our Embedding Model, we used a ResNet50 with weights="imagenet", our target shape + (3,) as input shape and top not included, followed by a DenseNet with 512 Units and relu activation, followed by Batch Normalization, followed by another DenseNet with 512 Units and relu activation, followed by a Batch Normalization and a DenseNet with 256 Units as output. Also we freeze the layers until the conv5_block1_out, which is important to avoid affecting the weights the model already learned. The bottom few layers are left trainable for fine tuning during their training.

### Distance Layer

The Distance layer is used to compute the Distance between our Triplets. It is a simplet layer which computes the distance between the anchor and the positive and the anchor and the negative image in the call function. It then returns the distance. 

### Siamese Network

A siamese network is a network consisting of 2 or more Networks which shari weights between each other. It is a common practice in Metric learning to use Siamese Networks, which is why we choose it. It contains a custom training and testing loop. Its output is a tuple containing the distance between anchor and negative as well as anchor and positive image. A function in the network then computes the loss by substracting the negative distance from the positive distance. 
We compile our network with the optimizer Adam and a learning rate of 

<!-- Training section -->
<a name='training'></a>

## Training

<!-- Visualization section -->
<a name='visualization'></a>

## Visualization


<!-- Results section -->
<a name='results'></a>

## Results
