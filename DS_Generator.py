"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import IMG_FOLDER, TARGET_SHAPE, IMG_CSV_SUBSET, IMG_CSV, MOST_COMMON_INDIVIDUALS


class DS_Generator():
    """
    Class used for dataset generation.
    
    Attributes
    ----------
    
    Methods
    ---------
    preprocess(subset=false)
        generates training and validation triplet dataset
    read_triplets(anchor, positive, negative)
        runs preprocess for anchor, positive, negative
    read_img(filename)
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
    """

    def __init__(self):
        """Constructor"""
        pass

    def preprocess(self, subset=False):
        """
        Generates Triplets as Tensorflow Datasets
        
        Parameters
        ----------
        subset : boolean, optional
            
        Returns
        -------
        train_ds: Tensorflow Take Dataset
            Dataset of Triplets for Training
        val_ds: Tensorflow Take Dataset
            Dataset of Triplets for Validation
        """
        df = IMG_CSV
        if subset:
            df = IMG_CSV_SUBSET

        anchors = np.empty(0)
        positives = np.empty(0)
        negatives = np.empty(0)

        for i in MOST_COMMON_INDIVIDUALS:
            pos_file_names = df[df.loc[:, "individual_id"] == i[0]].loc[:,
                             "image"].to_numpy()  # anchors + positives
            neg_file_names = df[df.loc[:, "individual_id"] != i[0]].loc[:,
                             "image"].to_numpy()  # negatives
            half_len = len(pos_file_names) // 2

            anchors = np.concatenate((anchors, pos_file_names[:half_len]))
            positives = np.concatenate((positives, pos_file_names[half_len:half_len * 2]))

            # random choice is not optimal (no guarantee that negative examples are not often times the same because of some bias)
            # we can look for a better solution later on
            negatives = np.concatenate((negatives, np.random.choice(neg_file_names, size=half_len, replace=False)))

        print("\nTriplet pairs generated! \n")

        img_count = len(anchors)

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
        
        Parameters
        ----------
        anchor: 
        
        positive
        
        negative
        
        Returns
        -------
        
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
        
        Parameters
        ----------
        filename: string
            filename of the File containing the Images
        Returns
        -------
        tensor: tensorflow tensor
            The Image as a Tensor
        """

        image_string = tf.io.read_file(IMG_FOLDER + filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE)
        tensor = tf.expand_dims(image, 0)

        return tensor
