import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import numpy as np

TARGET_SHAPE = (200, 200)

base_cnn = resnet.ResNet50(weights = "imagenet", input_shape = TARGET_SHAPE + (3,), include_top = False)
flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation = "relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation = "relu")(dense1)
dense1 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_clock1_out":
        trainable = True
    layer.trainable = trainable