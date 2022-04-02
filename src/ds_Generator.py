"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import TRAIN_DATA_PATH, BATCH_SIZE, TARGET_SHAPE, NUMBER_OF_SPECIES
import math
import random


def redo_counts(df):
    df = df.copy()
    df["species_counts"] = df.groupby('species_label')["species_label"].transform('count')
    df['individum_count'] = df.groupby('individual_id')['individual_id'].transform('count')
    return df


def triplet_loss_val_split(df, split_ratio, seed):
    # We only want to take away individums with more then 2 images,so we still can use them for triplet-loss training

    if split_ratio:
        values_with_more_then_2_instances_df = df[df["individum_count"] > 2]
        values_with_2_instances_df = df[df["individum_count"] == 2]

        classes_we_could_remove = {i: [] for i in values_with_more_then_2_instances_df['label']}
        indexes_we_cant_remove = list(values_with_2_instances_df.index)

        for index in values_with_more_then_2_instances_df.index:
            name = values_with_more_then_2_instances_df.loc[index, 'label']
            classes_we_could_remove[name].append(index)

        indexes_we_could_remove = list()

        for name in classes_we_could_remove:
            to_keep = classes_we_could_remove[name][:2]
            to_remove = classes_we_could_remove[name][2:]

            indexes_we_cant_remove.extend(to_keep)
            indexes_we_could_remove.extend(to_remove)

        # for every class throw away two training data

        # seed for replicability
        random.seed(seed)
        # shuffle indexes for randomness
        random.shuffle(indexes_we_could_remove)

        # get cut length
        cut = math.ceil(len(indexes_we_could_remove) * split_ratio)
        # indexes we want to keep
        keep_indexes = indexes_we_could_remove[cut:] + indexes_we_cant_remove
        # indexes for val ds
        val_indexes = indexes_we_could_remove[:cut]

        # index dfs with chosen indexes
        val_df = df.loc[val_indexes]
        train_df = df.loc[keep_indexes]

        # redo counts
        val_df = redo_counts(val_df)
        train_df = redo_counts(train_df)

        return train_df, val_df
    else:
        return df, None


def split_df_by_even_uneven(train_df, counts_column, label):
    # redo counts
    train_df = redo_counts(train_df)
    # split
    even_df = train_df[train_df[counts_column] % 2 == 0]
    uneven_df = train_df[train_df[counts_column] % 2 == 1]
    # get the indexes of the data-points with even/ uneven occurrences
    even_indices_list = list(even_df.index)
    # get the set of uneven classes
    set_of_uneven_classes = {a for a in uneven_df[label]}
    return even_df, uneven_df, even_indices_list, set_of_uneven_classes


def shuffle_container_order(train_df, amount_of_containers):
    """ This part is primarily for a good train,test,val split for our species data, in such a way that every dataset contains instances of all species.
    To achieve this more often we shuffle the order of containers:"""

    train_df = train_df.copy()

    last_container = amount_of_containers  # We do not want to shuffle the last one
    list_to_shuffle = list(range(0, amount_of_containers - 1))  # hence, the minus 1
    random.shuffle(list_to_shuffle)
    list_to_shuffle.append(last_container - 1)  # append the last container

    # reassign order
    train_df["assign_to"] = np.array([list_to_shuffle[int(i - 1)] for i in train_df["assign_to"]])
    return train_df


is_even = lambda x: x % 2 == 0


def smart_batches(df: pd.core.frame.DataFrame, BATCH_SIZE: int, task: str = "individual", seed=0,
                  val_split=0.1) -> pd.core.frame.DataFrame:
    """
    This is one of the most important functions:
    -----------------
    arguments:
    df - pandas data frame of our data
    seed - to generate same train/val split when reloading model
    BATCH_SIZE - the bath_sie of our tensorflow dataset, must be even
    task - either "individual_id" or "species", Specifies if we want to create train to identify species or individuals.
    val_split - whether you want some indiviudals to be split up vor validation purposes -> only implemented when task=indivual
    
    -----------------
    returns
    Ordered Data Frame for Tensorflow Data set creation, such that the batches are valid for the triplet loss,
    i.e. never contains only one positve.
    """
    assert task in ["individual",
                    "species"], 'task has to be either "individual_id" or "species"" and must be column index of df'

    assert is_even(BATCH_SIZE), "BATCH_SIZE must be even"

    # refresh counts just in case
    df = redo_counts(df)

    if task == "individual":
        label = "label"
        counts_column = "individum_count"
        df = df[df["individum_count"] > 1]
        # generate split
        train_df, val_df = triplet_loss_val_split(df, val_split, seed)

    elif task == "species":
        label = "species_label"
        counts_column = "species_counts"
        train_df = df

    # now we start working on the constraint problem
    # first we need to know the amount of containers
    amount_of_containers = math.ceil(len(train_df) / BATCH_SIZE)
    # then we make a numpy array, which holds for every container the amount of space
    container = np.zeros(amount_of_containers)
    container[:-1] = BATCH_SIZE
    # the last container has just the amount of data-points which are missing

    container[-1] = len(train_df) % BATCH_SIZE
    # assert that the last container has at least 2 places
    assert container[-1] >= 2, "A very unlikely case happened, try a val_split which is just a bit different or in " \
                               "case of species remove 2 random data-points from your df "

    # new column container assignment
    train_df["assign_to"] = np.nan

    # get dfs containing even/uneven values + indices of even datapoints + the set of uneven classes
    even_df, uneven_df, even_indices_list, set_of_uneven_classes = split_df_by_even_uneven(train_df, counts_column,
                                                                                           label)

    # make a dict with uneven_labels as keys and a empty list as value
    uneven_labels = {a: [] for a in uneven_df[label].array}

    # then assign every key all the indexes of the data-points which belong to it
    for index, int_label in zip(uneven_df.index, uneven_df[label].array):
        uneven_labels[int_label].append(index)

    # now we want to have only pairs of 3 datapoint for each uneven label
    # to achieve this we simply keep 3 and put the rest, which we know are always and even amount to the even indexes
    for int_label in uneven_labels:
        # if we have more then 3 data-points
        if len(uneven_labels[int_label]) > 3:
            # triplet we want to keep
            keep = uneven_labels[int_label][:3]
            # rest we want to put to the evenset
            rest = uneven_labels[int_label][3:]
            # put it there
            even_indices_list.extend(rest)
            # keep only triplet
            uneven_labels[int_label] = keep

    # now we create the list of uneven indices and shuffle it for randomness
    uneven_indices_list = [uneven_labels[a] for a in uneven_labels]
    random.shuffle(uneven_indices_list)

    # Now we have a delicate last problem
    # if we have an even amount of uneven classes and an even amount of space in the last container we do not have a problem
    if is_even(len(uneven_indices_list)) and is_even(container[-1]):
        pass

    # If we have an uneven amount of uneven classes and even amount of space in the last container
    # we just put a triplet in to the last container to make everything even
    elif not is_even(len(uneven_indices_list)) and not is_even(container[-1]):
        # we put 3 items in -> 3 space less
        container[-1] -= 3
        # get the first triplet
        first_triplet = uneven_indices_list.pop()
        # assign it to last container (we start counting with 0 like in a true pythonic fashion, hence the -1)
        train_df.loc[first_triplet, "assign_to"] = len(container) - 1

    # This case should never happen, if so something strange happened
    else:
        raise NameError('We made an error in the concept of the algorithm')


    # Now we should have only an even amount of triplet pairs
    assert is_even(len(uneven_indices_list)), "stf went horribly wrong"
    assert is_even(len(even_indices_list)), "stf went horribly wrong"

    # because it is even we now can generate the pairs 3+3 = 6 nicely by zipping + clever indexing
    combined_double_triplets = [a + b for a, b in zip(uneven_indices_list[::2], uneven_indices_list[1::2])]


    assert all([len(a) == 6 for a in combined_double_triplets])

    # beacause we assigned the some members of the uneven set to the even_indices_list we have to redo our even_df
    even_df = train_df.loc[even_indices_list]

    # no we want to form the double pairs
    # to to this we sort by label to then apply clever indexing (we can do this because we know that every label is represented an even amount of times in the df
    even_labels = even_df[label].sort_values().index
    # create pairs by indexing + zipping
    combined_even_doubles = [[a, b] for a, b in zip(even_labels[::2], even_labels[1::2])]
    # shuffle again for randomness
    random.shuffle(combined_even_doubles)

    # We check whether a tuple has the same label
    assert all([train_df.loc[a, label] == train_df.loc[b, label] for a, b in combined_even_doubles])

    # No all the hard work is done, and it is time to distribute our data-points to the container

    # small func to make the next part beautifuler
    next_step = lambda i: i + 1 if i + 1 != len(container) else 0

    # distribute the uneven triplets
    i = 0
    while combined_double_triplets:

        # if the container does not have enough space-> go to next
        if container[i] < 6:
            i = next_step(i)
            continue
        # get the first triplet pair
        triplets = combined_double_triplets.pop()
        # assign it
        train_df.loc[triplets, "assign_to"] = i
        # container has less space now
        container[i] -= 6

        i = next_step(i)

    # distribute the even triplets
    i = 0
    while combined_even_doubles:
        # if the container does not have enough space-> go to next
        if container[i] < 2:
            i = i + 1 if i + 1 != len(container) else 0
            continue

        double = combined_even_doubles.pop()
        container[i] -= 2
        train_df.loc[double, "assign_to"] = i

        i = i + 1 if i + 1 != len(container) else 0

    # all containers should be empty now
    assert np.all(container == 0)

    # This part is primarily for a good train,test,val split for our species data, in such a way that every dataset contains instances of all species
    # We shuffle the order of containers
    train_df = shuffle_container_order(train_df, amount_of_containers)

    # By sorting by the assignment-order we now achieve the good ordering for correct batches
    train_df = train_df.sort_values(["assign_to"])

    if task == "individual":
        return train_df, val_df
    elif task == "species":
        return train_df


class DS_Generator():
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

    def generate_species_data(self, df, factor_of_validation_ds=0.1, factor_of_test_ds=0.1, batch_size=None,
                              augment=False, seed=None, return_eval_data=False):
        global NUMBER_OF_SPECIES
        """This function creates the tensorflow dataset for training:
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training
        seed - to generate same train/val split when reloading model
        factor_of_validation_ds - float / between 0 and 1 -> Percentage auf validation dataset for splitup.
            Note: If we split increase the ds size via augmentation, the percentage will only be of the "real" data

        
        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        
        augment - Bool/ wether you want to apply data augmentaion
        return_eval_data - whether to return data for evaluation (test_ds + df)
        -----------------
        returns:
        train_ds,val_ds
        """

        # Asserts for function

        assert 0 <= factor_of_validation_ds <= 1, "Must be percentage"
        assert 0 <= factor_of_test_ds <= 1, "Must be percentage"

        if batch_size is None:
            batch_size = BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        df = smart_batches(df, batch_size, task="species", seed=seed)

        ds = self.build_ds(df["image"], df["species_label"])

        # one_hot encode labels
        ds = ds.map(lambda img, label: (img, tf.one_hot(label, NUMBER_OF_SPECIES)))

        df["which_set"] = "new_column"

        val_length = math.floor(factor_of_validation_ds * len(ds))
        val_ds = ds.take(val_length)
        df.iloc[range(val_length), -1] = "val_ds"

        test_length = math.floor(factor_of_test_ds * len(ds))
        test_ds = ds.take(test_length)
        df.iloc[range(val_length, val_length + test_length), -1] = "test_ds"

        train_ds = ds.skip(val_length + test_length)
        df.iloc[range(val_length + test_length, len(ds)), -1] = "train_ds"

        if augment:
            train_ds = train_ds.map(self.augment, num_parallel_calls=8)

        train_ds = train_ds.batch(batch_size).prefetch(10)
        val_ds = val_ds.batch(batch_size).prefetch(10)

        if not return_eval_data:
            return train_ds, val_ds
        elif return_eval_data:
            return train_ds, val_ds, test_ds, df

    def generate_individual_data(self, df, augment=False, batch_size=None, seed=None, val_split=0.1, return_eval_data=False):
        """This function creates the tensorflow dataset for training:
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training
        seed - to generate same train/val split when reloading model
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

        if batch_size is None:
            batch_size = BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        # Create order for the batches
        train_df, val_df = smart_batches(df, batch_size, task="individual", seed=seed, val_split=val_split)

        train_ds = self.build_ds(train_df["image"], train_df["label"])
        if augment:
            train_ds = train_ds.map(self.augment)
        train_ds = train_ds.batch(batch_size).prefetch(10)

        val_ds = self.build_ds(val_df["image"], val_df["label"])
        val_ds = val_ds.batch(batch_size).prefetch(10)

        if return_eval_data == False:
            return train_ds
        elif return_eval_data:
            return train_ds, val_ds, train_df, val_df

    def build_ds(self, imgage_paths, classes):
        image_paths = TRAIN_DATA_PATH + "/" + imgage_paths
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(classes, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(self.prepare_images_mapping, num_parallel_calls=8)
        return ds

    def generate_single_individuals_ds(self,df,batch_size):
        df = df[df["individum_count"]==1]
        return self.build_ds(df["image"], df["label"]).batch(batch_size).prefetch(10), df