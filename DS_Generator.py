"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES


class DS_Generator():
    """
    Class used for dataset generation.
    
    Attributes
    ----------
    
    Methods
    ---------
    """

    def __init__(self):
        global PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES
        self.folder_of_data = PATH_FOR_OUR_TRAINING_DATA
        self.limit = UPPER_LIMIT_OF_IMAGES

    def generate(self,augment=False):
        """
        args:
        
        augment - bool / Wether you want data augmentation until upper limit"""

        df = pd.read_csv(self.folder_of_data + "/data.csv", index_col=0)

        image_paths = self.folder_of_data + "/" + df["image"]
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(df["label"], dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        func = self.prepare_images_mapping
        ds = ds.map(func, num_parallel_calls=8)
        ds = ds.shuffle(1024).batch(32)

        return ds

    def prepare_images_mapping(self,path, label):
        x = tf.io.read_file(path)
        x = tf.image.decode_jpeg(x,3)
        x = tf.cast(x, dtype=tf.float32)
        x *= (2 / 255)
        x -= 1
        return x,label
