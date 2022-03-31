import sys
sys.path.append("..")
sys.path.append("../src")
import tensorflow as tf
from util import TRAIN_SPECIES_DF, INDIVIDUMS_SEED
import tensorflow_addons as tfa

SOFT_MAX_MODEL_PATH = "../Saved Models/inception_v3_max_pooling_imagenet_weights"


def return_new_siamese_model(load_weitghs_path = None):
    Input = tf.keras.Input((224,224,3))
    base = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_tensor=Input,
        pooling="max"
    )
    dense1 = tf.keras.layers.Dense(512, activation="relu")(base.output)  # chop of last layer
    dense1 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_1")(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_2")(dense2)
    dense3 = tf.keras.layers.Dense(256)(dense2)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense3)
    model = tf.keras.Model(inputs=Input, outputs=output, name="SiameseNetwork_untrained")

    if load_weitghs_path is None:
        pass
    else:
        model.load_weights(load_weitghs_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())

    return model


def return_soft_max_pretrained_siamese_model(load_weitghs_path = None):
    old_model = tf.keras.models.load_model(SOFT_MAX_MODEL_PATH)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(old_model.layers[-2].output)  # chop of last layer
    dense1 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_1")(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_2")(dense2)
    dense3 = tf.keras.layers.Dense(256)(dense2)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense3)
    model = tf.keras.Model(inputs=old_model.input, outputs=output, name="SiameseNetworkwithsoftmaxweights")

    if load_weitghs_path is None:
        pass
    else:
        model.load_weights(load_weitghs_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())

    return model