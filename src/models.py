import tensorflow as tf
import util
import tensorflow_addons as tfa
import os

SOFT_MAX_MODEL_PATH = "Saved Models/inception_v3_max_pooling_imagenet_weights"


def embedding_part(Input):
    dense1 = tf.keras.layers.Dense(512, activation="relu")(Input)  # chop of last layer
    dense1 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_1")(dense1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dense1)
    dense2 = tf.keras.layers.BatchNormalization(name="Embedding_BatchNormalization_2")(dense2)
    dense3 = tf.keras.layers.Dense(256)(dense2)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense3)
    return output


def load_weights_and_compile(model, load_weights_path):
    if load_weights_path is None:
        pass
    else:
        model.load_weights(load_weights_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())

    if model.name not in os.listdir(util.SAVING_PATH):
        os.makedirs(util.SAVING_PATH + model.name)
        os.makedirs(util.SAVING_PATH + model.name + "/logs")
        os.makedirs(util.SAVING_PATH + model.name + "/saves")
    return model


def return_siamese_control_model(load_weights_path=None):
    Input = tf.keras.Input((224, 224, 3))
    base = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=Input,
        pooling="max"
    )
    output = embedding_part(base.output)  # chop of last layer
    model = tf.keras.Model(inputs=Input, outputs=output, name="ControlSiameseNetwork")

    return load_weights_and_compile(model, load_weights_path)


def return_new_siamese_model(load_weights_path=None):
    Input = tf.keras.Input((224, 224, 3))
    base = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_tensor=Input,
        pooling="max"
    )
    output = embedding_part(base.output)  # chop of last layer
    model = tf.keras.Model(inputs=Input, outputs=output, name="SiameseNetwork_untrained")

    return load_weights_and_compile(model, load_weights_path)


def return_soft_max_pretrained_siamese_model(load_weights_path=None):
    old_model = tf.keras.models.load_model(SOFT_MAX_MODEL_PATH)

    output = embedding_part(old_model.layers[-2].output)  # chop of last layer
    model = tf.keras.Model(inputs=old_model.input, outputs=output, name="SiameseNetworkwithsoftmaxweights")

    return load_weights_and_compile(model, load_weights_path)
