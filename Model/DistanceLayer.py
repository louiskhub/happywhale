"""
Distance Layer Model: Calculates the Distance between the images. 

Louis Kapp, Felix Hammer, Yannik Ullrich
"""
happywhale = __import__("happywhale-main")

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
from happywhale.util import TARGET_SHAPE

# Base cnn: Resnet followed by Flattening, Dense and Batch normalization 
base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=TARGET_SHAPE + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

# Embedding Model 
embedding = Model(base_cnn.input, output, name="Embedding")

# Freezing weights for the first few layers
trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable


class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        anchor, positive, negative = self.embed(anchor, positive, negative)
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

    def embed(self, anchor, positive, negative):
        return(
            embedding(resnet.preprocess_input(anchor)),
            embedding(resnet.preprocess_input(positive)),
            embedding(resnet.preprocess_input(negative))
        )


anchor_input = layers.Input(name="anchor", shape=TARGET_SHAPE + (3,))
positive_input = layers.Input(name="positive", shape=TARGET_SHAPE + (3,))
negative_input = layers.Input(name="negative", shape=TARGET_SHAPE + (3,))

distances = DistanceLayer()(anchor_input, positive_input, negative_input)

siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)
