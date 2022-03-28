"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import TRAIN_DATA_PATH, BATCH_SIZE, TARGET_SHAPE
import math
import random
from src.data_augmentation import extract_foreground


def df_filter_for_indidum_training(train_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    ids, respective_counts = np.unique(train_df["individual_id"].values, return_counts=True)
    ids = ids[respective_counts > 1]  # boolean index the ids

    respective_counts = respective_counts[respective_counts > 1]  # and counts for later

    filter_function = np.vectorize(lambda x: x in ids)  # our filter function

    train_df = train_df.iloc[filter_function(train_df["individual_id"])]  # filter df

    train_df.index = range(len(train_df))  # reindex
    return train_df


def smart_batches(df: pd.core.frame.DataFrame, BATCH_SIZE: int, task: str = "individual", create_val_df = False) -> pd.core.frame.DataFrame:
    """
    This is one of the most important functions:
    -----------------
    arguments:
    df - pandas data frame of our data
    BATCH_SIZE - the bath_sie of our tensorflow dataset, must be even
    task - either "individual_id" or "species", Specifies if we want to create train to identify species or individuals.
    create_val_df - whether you want some indiviudals to be split up vor validation purposes -> only implemented when task=indivual
    
    -----------------
    returns
    Ordered Data Frame for Tensorflow Data set creation, such that the batches are valid for the triplet loss,
    i.e. never contains only one positve.
    """
    assert task in ["individual",
                    "species"], 'task has to be either "individual_id" or "species"" and must be column index of df'

    if create_val_df:
        assert task == "individual", "only implemented when task=indivual"

    if task == "individual":
        label = "label"
        counts_column = "individum_count"
        df = df_filter_for_indidum_training(df)
    elif task == "species":
        label = "species_label"
        counts_column = "species_counts"
    df = df.copy()
    assert BATCH_SIZE % 2 == 0, "BATCH_SIZE must be even"

    # refresh counts just in case
    df["species_counts"] = df.groupby('species_label')["species_label"].transform('count')
    df['individum_count'] = df.groupby('individual_id')['individual_id'].transform('count')

    if create_val_df:
        indexes_we_could_remove = list(df[df["individum_count"] > 2].index)
        random.shuffle(indexes_we_could_remove)

        split_ratio = 0.05
        cut = int(len(indexes_we_could_remove)*split_ratio)
        keep_indexes = indexes_we_could_remove[cut:]
        val_indexes = indexes_we_could_remove[:cut]
        val_df = df.loc[val_indexes]
        df = df.loc[keep_indexes]

        # again, redo counts
        df["species_counts"] = df.groupby('species_label')["species_label"].transform('count')
        df['individum_count'] = df.groupby('individual_id')['individual_id'].transform('count')
    else:
        val_df = None

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

    return df.sort_values(["assign_to"]),val_df



class DataSet_Generator():
    def __init__(self):
        pass

    def prepare_images_mapping(self, path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img *= (2 / 255)
        img -= 1
        return img, label

    def augment(self, img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.70, 1.30)
        img = tf.image.random_contrast(img, 0.80, 1.20)
        img = tf.image.random_brightness(img, 0.10)
        return img, label

    def generate_species_data(self, df, factor_of_validation_ds=0.1, batch_size=None, augment=False):
        global TARGET_SHAPE
        """This function creates the tensorflow dataset for training:
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training

        factor_of_validation_ds - float / between 0 and 1 -> Percentage auf validation dataset for splitup.
            Note: If we split increase the ds size via augmentation, the percentage will only be of the "real" data

        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        
        augment - Bool/ wether you want to apply data augmentaion
        -----------------
        returns:
        train_ds,val_ds
        """

        # Asserts for function

        assert 0 <= factor_of_validation_ds <= 1, "Must be percentage"

        if batch_size is None:
            batch_size = BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        df, _ = smart_batches(df, batch_size, "species")


        image_paths = TRAIN_DATA_PATH + "/" + df["image"]

        number_of_classes = len(set(df["species_label"]))

        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(df["species_label"], dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # map preprosessing
        ds = ds.map(self.prepare_images_mapping , num_parallel_calls=8)

        #one_hot encode labels
        ds = ds.map(lambda img,label : (img, tf.one_hot(label, number_of_classes)))
        if factor_of_validation_ds > 0:
            length = math.floor(factor_of_validation_ds * len(ds))
            val_ds = ds.take(length)
            train_ds = ds.skip(length)
        else:
            val_ds = None
            train_ds = ds
            print("No validation set wanted, hence we will return None")

        if augment:
            train_ds = train_ds.map(self.augment, num_parallel_calls=8)

        train_ds = train_ds.batch(batch_size).prefetch(10)
        val_ds = val_ds.batch(batch_size).prefetch(10)

        ds = ds.batch(batch_size)

        return train_ds,val_ds

    def generate_individual_data(self, df, increase_ds_factor=1,batch_size=None,with_val_ds=False):

        global TARGET_SHAPE
        """This function creates the tensorflow dataset for training:
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training

        increase_ds_factor - int / either 1,2,3 -> By with factor do you want to increase dataset via augmentaion
            1 -> keep size, no change
            2 -> double ds size via augment1 function
            3 -> triple ds size via augment1 + augment2 function

        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        with_val_ds - Split apart a small ds for accuracy estimations
        -----------------
        returns:
        train_ds,val_ds
        """

        # Asserts for function

        assert increase_ds_factor in [1, 2, 3], "Not supported value"

        if batch_size is None:
            batch_size = BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        # Create order for the batches
        df, val_df = smart_batches(df, batch_size, "individuals",with_val_ds)

        image_paths = TRAIN_DATA_PATH + "/" + df["image"]

        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)

        labels = tf.convert_to_tensor(df["label"], dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # map preprosessing
        ds = ds.map(self.prepare_images_mapping, num_parallel_calls=8)
        ds = ds.batch(batch_size)

        if val_df is not None:
            image_paths = TRAIN_DATA_PATH + "/" + val_df["image"]
            image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
            labels = tf.convert_to_tensor(val_df["label"], dtype=tf.int32)
            val_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            val_ds = val_ds.map(self.prepare_images_mapping, num_parallel_calls=8)
        else:
            val_ds = None

        #if increase_ds_factor == 1:
        #    pass
        #elif increase_ds_factor == 2:
        #    augmented_ds1 = ds.map(self.augment1, num_parallel_calls=8)
        #    train_ds = ds.concatenate(augmented_ds1)
        #elif increase_ds_factor == 3:
        #    augmented_ds1 = ds.map(self.augment1, num_parallel_calls=8)
        #    augmented_ds2 = ds.map(self.augment2, num_parallel_calls=8)#
#
 #           train_ds = train_ds.concatenate(augmented_ds1)
#          train_ds = train_ds.concatenate(augmented_ds2)

        # Finally, batch train_ds
        return ds,val_ds

    # for now code leichen aber vielleicht später
    def augment1(self, x, label):
        x = tf.image.random_crop(x, TARGET_SHAPE + (1,))
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x, label

    def augment2(self, x, label):
        x = tf.image.random_contrast(x, 0.2, 0.5)
        x = tf.image.random_brightness(x, 0.2)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_flip_left_right(x)
        return x, label

