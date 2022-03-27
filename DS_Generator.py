"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES, BATCH_SIZE, training_df, TARGET_SHAPE
import math
import random


def df_filter_for_indidum_training(train_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    ids, respective_counts = np.unique(train_df["individual_id"].values, return_counts=True)
    ids = ids[respective_counts > 1]  # boolean index the ids

    respective_counts = respective_counts[respective_counts > 1]  # and counts for later

    filter_function = np.vectorize(lambda x: x in ids)  # our filter function

    train_df = train_df.iloc[filter_function(train_df["individual_id"])]  # filter df

    train_df.index = range(len(train_df))  # reindex
    return train_df


def smart_batches(df: pd.core.frame.DataFrame, BATCH_SIZE: int, task: str = "individual") -> pd.core.frame.DataFrame:
    """
    This is one of the most important functions:
    -----------------
    arguments:
    df - pandas data frame of our data
    BATCH_SIZE - the bath_sie of our tensorflow dataset, must be even
    task - either "individual_id" or "species", Specifies if we want to create train to identify species or individuals.
    
    -----------------
    returns
    Ordered Data Frame for Tensorflow Data set creation, such that the batches are valid for the triplet loss,
    i.e. never contains only one positve.
    """
    assert task in ["individual",
                    "species"], 'task has to be either "individual_id" or "species"" and must be column index of df'
    if task == "individual":
        label = "label"
        counts_column = "individum_count"
        df = df_filter_for_indidum_training(df)
    elif task == "species":
        label = "species_label"
        counts_column = "species_counts"
    df = df.copy()

    df["species_counts"] = df.groupby('species_label')["species_label"].transform('count')
    df['individum_count'] = df.groupby('individual_id')['individual_id'].transform('count')

    assert BATCH_SIZE % 2 == 0, "BATCH_SIZE must be even"

    df["assign_to"] = np.nan

    even_mask = (df[counts_column] % 2 == 0).array
    uneven_mask = np.logical_not(even_mask)

    even_indices_list = list(df[even_mask].index)
    uneven_df = df[uneven_mask]

    amount_of_containers = math.ceil(len(df) / BATCH_SIZE)
    container = np.array([BATCH_SIZE for i in range(amount_of_containers - 1)] + [len(df) % BATCH_SIZE])

    set_of_uneven_classes = {a for a in uneven_df[label]}

    if not len(set_of_uneven_classes) % 2 != container[-1] % 2:
        unlucky_class = random.choice(uneven_df.index)
        df.drop(index=unlucky_class)

        even_mask = (df[counts_column] % 2 == 0).array
        uneven_mask = np.logical_not(even_mask)

        even_indices_list = list(df[even_mask].index)
        uneven_df = df[uneven_mask]
        set_of_uneven_classes = {a for a in uneven_df[label]}

        print(f"We threw away the datapoint with index {unlucky_class} ")

    uneven_labels = {a: [] for a in uneven_df[label].array}
    for index, int_label in zip(uneven_df.index, uneven_df[label].array):
        uneven_labels[int_label].append(index)

    for int_label in uneven_labels:
        if len(uneven_labels[int_label]) > 3:
            rest = uneven_labels[int_label][3:]
            keep = uneven_labels[int_label][:3]
            even_indices_list.extend(rest)

            uneven_labels[int_label] = keep

    uneven_indices_list = [uneven_labels[a] for a in uneven_labels]
    random.shuffle(uneven_indices_list)

    if len(set_of_uneven_classes) % 2 == 1:
        container[-1] -= 3
        first_triplet = uneven_indices_list.pop()
        df.loc[first_triplet, "assign_to"] = len(container) - 1
    assert len(uneven_indices_list) % 2 == 0, "stf went horbly wrong"

    combined_double_triplets = [a + b for a, b in zip(uneven_indices_list[::2], uneven_indices_list[1::2])]
    assert all([len(a) == 6 for a in combined_double_triplets])

    even_df = df.loc[even_indices_list]
    even_labels = even_df[label].sort_values().index

    combined_even_doubles = [[a, b] for a, b in zip(even_labels[::2], even_labels[1::2])]
    random.shuffle(combined_even_doubles)

    assert all([df.loc[a, label] == df.loc[b, label] for a, b in combined_even_doubles])
    i = 0
    while combined_double_triplets:

        if container[i] < 6:
            i = i + 1 if i + 1 != len(container) else 0
            continue

        triplets = combined_double_triplets.pop()
        container[i] -= 6
        df.loc[triplets, "assign_to"] = i

        i = i + 1 if i + 1 != len(container) else 0

    i = 0
    while combined_even_doubles:

        if container[i] < 2:
            i = i + 1 if i + 1 != len(container) else 0
            continue

        double = combined_even_doubles.pop()
        container[i] -= 2
        df.loc[double, "assign_to"] = i

        i = i + 1 if i + 1 != len(container) else 0

    assert np.all(container == 0)

    return df.sort_values(["assign_to"])


class DS_Generator():
    """
    Class used for dataset generation.
    
    Attributes
    ----------
    
    Methods
    ---------
    """

    def __init__(self):
        pass

    def generate(self, df, factor_of_validation_ds=0.1, increase_ds_factor=1, individuals=False, batch_size=None, one_hot_encode = False):
        global TARGET_SHAPE
        """This function creates the tensorflow dataset for training:
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training

        factor_of_validation_ds - float / between 0 and 1 -> Percentage auf validation dataset for splitup.
            Note: If we split increase the ds size via augmentation, the percentage will only be of the "real" data

        increase_ds_factor - int / either 1,2,3 -> By with factor do you want to increase dataset via augmentaion
            1 -> keep size, no change
            2 -> double ds size via augment1 function
            3 -> triple ds size via augment1 + augment2 function

        individuals - bool / whether you want to try identifying individuals or species

        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        
        one_hot_encode - Bool / whether to one hot encode the labels

        -----------------
        returns:
        train_ds,val_ds
        """

        # Asserts for function

        assert 0 <= factor_of_validation_ds <= 1, "Must be percentage"
        assert increase_ds_factor in [1, 2, 3], "Not supported value"

        if batch_size is None:
            batch_size = BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        if individuals:
            task = "individuals"
        else:
            task = "species"

        # Create order for the batches
        df = smart_batches(df, batch_size, task)

        image_paths = PATH_FOR_OUR_TRAINING_DATA + "/" + df["image"]

        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)

        # choose the wright labes according to task
        if individuals:
            labels = tf.convert_to_tensor(df["label"], dtype=tf.int32)
        else:
            labels = tf.convert_to_tensor(df["species_label"], dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # map preprosessing

        num_classes = len({i for i in df["species_label"]})
        if one_hot_encode:
            func = lambda img,label : (img, tf.one_hot(label,num_classes))
            ds = ds.map(func,num_parallel_calls=8)

        ds = ds.map(self.prepare_images_mapping, num_parallel_calls=8)

        # split up validation set
        if factor_of_validation_ds > 0:
            length = math.floor(factor_of_validation_ds * len(ds))
            val_ds = ds.take(length)
            val_ds = val_ds.batch(batch_size)
            train_ds = ds.skip(length)
        else:
            val_ds = None
            train_ds = ds
            print("No validation set wanted, hence we will return None")

        if increase_ds_factor == 1:
            pass
        elif increase_ds_factor == 2:
            augmented_ds1 = train_ds.map(self.augment1, num_parallel_calls=8)
            train_ds = train_ds.concatenate(augmented_ds1)
        elif increase_ds_factor == 3:
            augmented_ds1 = train_ds.map(self.augment1, num_parallel_calls=8)
            augmented_ds2 = train_ds.map(self.augment2, num_parallel_calls=8)

            train_ds = train_ds.concatenate(augmented_ds1)
            train_ds = train_ds.concatenate(augmented_ds2)

        # Finally, batch train_ds
        train_ds = train_ds.batch(batch_size)

        return train_ds, val_ds

    def prepare_images_mapping(self, path, label):
        x = tf.io.read_file(path)
        x = tf.image.decode_jpeg(x, channels=1)
        x = tf.cast(x, dtype=tf.float32)
        x *= (2 / 255)
        x -= 1
        return x, label

    def augment1(self, x, label):
        x = tf.image.random_crop(x, TARGET_SHAPE + (1,))
        x = tf.image.resize(x, TARGET_SHAPE)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x, label

    def augment2(self, x, label):
        x = tf.image.random_contrast(x, 0.2, 0.5)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x, label
