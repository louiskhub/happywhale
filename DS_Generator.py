"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES, BATCH_SIZE


class DS_Generator():
    """
    Class used for dataset generation.
    
    Attributes
    ----------
    
    Methods
    ---------
    """

    def __init__(self):
        global PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES, BATCH_SIZE
        self.folder_of_data = PATH_FOR_OUR_TRAINING_DATA
        self.limit = UPPER_LIMIT_OF_IMAGES
        self.batch_size = BATCH_SIZE

    def generate(self,df,augment=False,individuals=False):
        """
        args:
        
        augment - bool / Wether you want data augmentation until upper limit"""

        image_paths = self.folder_of_data + "/" + df["image"]
        
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        
        if individuals:
            labels = tf.convert_to_tensor(df["label"], dtype=tf.int32)
        else:
            labels = tf.convert_to_tensor(df["species_label"], dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        func = self.prepare_images_mapping
        ds = ds.map(func, num_parallel_calls=8)
        ds = ds.batch(self.batch_size)

        return ds

    def prepare_images_mapping(self,path, label):
        x = tf.io.read_file(path)
        x = tf.image.decode_jpeg(x,3)
        x = tf.cast(x, dtype=tf.float32)
        x *= (2 / 255)
        x -= 1
        return x,label
