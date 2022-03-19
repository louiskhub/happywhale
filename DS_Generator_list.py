import tensorflow as tf
import numpy as np
from random import sample
import pandas as pd
from util import IMG_FOLDER, TARGET_SHAPE, IMG_CSV_SUBSET, MOST_COMMON_INDIVIDUALS


class DS_Generator():

    def __init__(self):
        pass

    def preprocess(self, individuals_path=None):

        anchors = []
        positives = []
        negatives = []

        for i in MOST_COMMON_INDIVIDUALS:
            pos_file_names = IMG_CSV_SUBSET[IMG_CSV_SUBSET.loc[:, "individual_id"] == i[0]].loc[:, "image"].tolist()  # anchors + positives
            neg_file_names = IMG_CSV_SUBSET[IMG_CSV_SUBSET.loc[:, "individual_id"] != i[0]].loc[:, "image"].tolist()  # negatives
            half_len = len(pos_file_names) // 2

            anchors = anchors + pos_file_names[:half_len]
            positives = positives + pos_file_names[half_len:half_len * 2]

            # random choice is not optimal (no guarantee that negative examples are not often times the same because of some bias)
            # we can look for a better solution later on
            n = sample(neg_file_names, k=half_len)
            negatives = negatives + n

        print("\n Triplet pairs generated! \n")

        img_count = len(anchors)

        # anchors = anchors.astype(np.str_)
        # positives = positives.astype(np.str_)
        # negatives = negatives.astype(np.str_)
        # anchors = tf.convert_to_tensor(anchors)
        # positives = tf.convert_to_tensor(positives)
        # negatives = tf.convert_to_tensor(negatives)

        anchor_ds = tf.data.Dataset.from_tensor_slices(anchors)
        positive_ds = tf.data.Dataset.from_tensor_slices(positives)
        negative_ds = tf.data.Dataset.from_tensor_slices(negatives)
        ds = tf.data.Dataset.zip((anchor_ds, positive_ds, negative_ds))

        print("Dataset generated! \n")

        ds = ds.map(self.read_triplets)

        # split
        train_ds = ds.take(round(img_count * 0.8))
        val_ds = ds.skip(round(img_count * 0.8))

        train_ds.shuffle(512).batch(5).prefetch(8)
        val_ds.shuffle(512).batch(5).prefetch(8)

        return train_ds, val_ds

    def read_triplets(self, anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """

        return (
            self.read_img(anchor),
            self.read_img(positive),
            self.read_img(negative),
        )

    def read_img(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        print(type(IMG_FOLDER))
        print(filename)
        image_string = tf.io.read_file(IMG_FOLDER + filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE)

        return image
