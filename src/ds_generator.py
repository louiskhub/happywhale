"""
Create a Tensorflow dataset and Triplets for the Triplet loss function.
@authors: fhammer, lkapp
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import util
import math
import random


def redo_counts(df):
    """
    Append count-columns for individuals and species to dataframe.
    -----------------
    arguments:
    df - pandas.DataFrame which needs the counts
    -----------------
    returns:
    df - pandas.DataFrame which has the counts
    """

    df = df.copy()
    df["species_counts"] = df.groupby('species_label')["species_label"].transform('count')
    df['individual_counts'] = df.groupby('individual_id')['individual_id'].transform('count')
    return df  # I got what u need


def triplet_loss_val_split(df, val_split_ratio, seed):
    """
    Splits the dataset into training and validation if val_split_ratio != 0.
    -----------------
    arguments:
    df - pandas.DataFrame which needs to be split
    val_split_ratio - Ratio of the validation set
    seed - seed for reproducible results
    -----------------
    returns:
    Train and Validation pandas.DataFrame if val_split_ratio != 0. Otherwise only Train DataFrame.
    """

    if val_split_ratio:
        
        # We only want to take away individuals with more than 2 images
        # this way we still can use them for triplet-loss training
        values_with_more_then_2_instances_df = df[df["individual_counts"] > 2]
        values_with_2_instances_df = df[df["individual_counts"] == 2]

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

        # for every class throw away two training data points

        # seed for reproducibility
        random.seed(seed)
        # shuffle indexes for randomness
        random.shuffle(indexes_we_could_remove)

        # get cut length
        cut = math.ceil(len(indexes_we_could_remove) * val_split_ratio)
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


def split_df_by_even_uneven(train_df, counts_column, label_column):
    """
    Split pandas.DataFrame into species/individuals with an even number of images 
    and species/individuals with an uneven number of images.
    -----------------
    arguments:
    train_df - pandas.DataFrame to be split
    counts_column - column which contains the interesting counts (species/individual)
    label_column - colum which contains the interesting labels (species/individual)
    -----------------
    returns:
    1. pandas.DataFrame containing all even species/individuals
    2. pandas.DataFrame containing all uneven species/individuals
    3. List containing all indices of even species/individuals
    4. Set containing all uneven species/individuals
    """

    # redo counts
    train_df = redo_counts(train_df)
    # split
    even_df = train_df[train_df[counts_column] % 2 == 0]
    uneven_df = train_df[train_df[counts_column] % 2 == 1]
    # get the indexes of the data-points with even/ uneven occurrences
    even_indices_list = list(even_df.index)
    # get the set of uneven classes
    set_of_uneven_classes = {a for a in uneven_df[label_column]}
    
    return even_df, uneven_df, even_indices_list, set_of_uneven_classes


def shuffle_batches_order(train_df, number_of_batches):
    """
    Ensure a good train, validation and test split for our species data.
    In such a way that every dataset contains instances of all species.
    -----------------
    arguments:
    train_df - pandas.DataFrame for training
    number_of_batches - number of batches
    -----------------
    returns:
    pandas.DataFrame for training with shuffled batches
    """

    train_df = train_df.copy()

    last_container = number_of_batches  # We do not want to shuffle the last one
    list_to_shuffle = list(range(0, number_of_batches - 1))  # hence, the minus 1
    random.shuffle(list_to_shuffle)
    list_to_shuffle.append(last_container - 1)  # append the last container

    # reassign order
    train_df["assign_to"] = np.array([list_to_shuffle[int(i - 1)] for i in train_df["assign_to"]])
    
    return train_df


def smart_batches(df: pd.core.frame.DataFrame, batch_size: int, task: str = "individual", seed=0,
                  val_split=0.1) -> pd.core.frame.DataFrame:
    """
    smart batch generation for SemiHardTripletLoss
    -----------------
    arguments:
    df - pandas.DataFrame of our data
    batch_size - the batch size of our tensorflow dataset (must be even)
    seed - get reproducible results
    task - either "individual" or "species"
    val_split - validation split (only implemented when task=individual)

    -----------------
    returns
    Ordered pandas.DataFrame for Tensorflow Data set creation.
    Ensures that the batches are valid for the triplet loss.
    (i.e. never contain only one positive)
    """

    assert task in ["individual",
                    "species"], 'task has to be either "individual_id" or "species"" and must be column index of df'

    assert util.IS_EVEN(batch_size), "BATCH_SIZE must be even"

    # refresh counts just in case
    df = redo_counts(df)

    if task == "individual":
        label = "label"
        counts_column = "individual_counts"
        df = df[df["individual_counts"] > 1]
        # generate split
        train_df, val_df = triplet_loss_val_split(df, val_split, seed)

    elif task == "species":
        label = "species_label"
        counts_column = "species_counts"
        train_df = df

    # now we start working on the constraint problem
    # first we need to know the amount of containers
    amount_of_batches = math.ceil(len(train_df) / batch_size)
    # then we make a numpy array, which holds for every container the amount of space
    container = np.zeros(amount_of_batches)
    container[:-1] = batch_size
    # the last container has just the amount of data-points which are missing

    container[-1] = len(train_df) % batch_size
    # assert that the last container has at least 2 places
    assert container[-1] >= 2, "A very unlikely case happened, try a val_split which is just a bit different or in " \
                               "case of species remove 2 random data-points from your df "

    # new column container assignment
    train_df["assign_to"] = np.nan

    # get dfs containing even/uneven values + indices of even datapoints + the set of uneven classes
    even_df, uneven_df, even_indices_list, set_of_uneven_classes = split_df_by_even_uneven(train_df,
                                                                                           counts_column,
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
    # if we have an even amount of uneven classes and an even amount of space in the last container
    # we do not have a problem
    if util.IS_EVEN(len(uneven_indices_list)) and util.IS_EVEN(container[-1]):
        pass

    # If we have an uneven amount of uneven classes and even amount of space in the last container
    # we just put a triplet in to the last container to make everything even
    elif not util.IS_EVEN(len(uneven_indices_list)) and not util.IS_EVEN(container[-1]):
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
    assert util.IS_EVEN(len(uneven_indices_list)), "stf went horribly wrong"
    assert util.IS_EVEN(len(even_indices_list)), "stf went horribly wrong"

    # because it is even we now can generate the pairs 3+3 = 6 nicely by zipping + clever indexing
    combined_double_triplets = [a + b for a, b in zip(uneven_indices_list[::2], uneven_indices_list[1::2])]

    assert all([len(a) == 6 for a in combined_double_triplets])

    # because we assigned the same members of the uneven set to the even_indices_list we have to redo our even_df
    even_df = train_df.loc[even_indices_list]

    # no we want to form the double pairs
    # to do this we sort by label to then apply clever indexing
    # (we can do this because we know that every label is represented an even amount of times in the df)
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

    # This part is primarily for a good train, test,val split for our species data,
    # in such a way that every dataset contains instances of all species
    # We shuffle the order of containers
    train_df = shuffle_batches_order(train_df, amount_of_batches)

    # By sorting by the assignment-order we now achieve the good ordering for correct batches
    train_df = train_df.sort_values(["assign_to"])

    if task == "individual":
        return train_df, val_df
    elif task == "species":
        return train_df


class DS_Generator():
    """ The class to be called for dataset generation. """

    def __init__(self):
        pass

    def prepare_images_mapping(self, path, label):
        """
        basic preprocessing and normalization steps
        -----------------
        arguments:
        path - absolute filepath of the image to be read
        label - label of the image to be read
        -----------------
        returns
        Preprocessed image and its label.
        """
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img *= (2 / 255)
        img -= 1
        return img, label

    def augment(self, img, label):
        """
        Basic randomized data augmentation steps.
        This function is seperated because we only want to augment our training data and not test/validation data.
        -----------------
        arguments:
        img - image to be augmented
        label - label of the image to be augmented
        -----------------
        returns
        Augmented image and its label.
        """
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.70, 1.30)
        img = tf.image.random_contrast(img, 0.80, 1.20)
        img = tf.image.random_brightness(img, 0.10)
        return img, label

    def generate_species_data(self, df, factor_of_validation_ds=0.1, factor_of_test_ds=0.1, batch_size=None,
                              augment=False, seed=None, return_eval_data=False):
        """
        This function creates the tensorflow dataset for training on species.
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training
        factor_of_validation_ds - float / between 0 and 1 -> Percentage auf validation dataset for splitup.
        factor_of_test_ds - float / between 0 and 1 -> Percentage auf test dataset for splitup.
        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        augment - Bool/ whether you want to apply data augmentation
        seed - to generate same train/val split when reloading model
        return_eval_data - whether to return data for evaluation (test_ds + df)
        -----------------
        returns:
        Training and Validation Dataset.
        """

        # Asserts for function

        assert 0 <= factor_of_validation_ds <= 1, "Must be percentage"
        assert 0 <= factor_of_test_ds <= 1, "Must be percentage"

        if batch_size is None:
            batch_size = util.BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        df = smart_batches(df, batch_size, task="species", seed=seed)

        ds = self.build_ds(df["image"], df["species_label"])

        # one_hot encode labels
        ds = ds.map(lambda img, label: (img, tf.one_hot(label, util.NUMBER_OF_SPECIES)))

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

    def generate_individual_data(self, df, augment=False, batch_size=None, seed=None, val_split=0.1,
                                 return_eval_data=False):
        """
        This function creates the tensorflow dataset for training on individuals.
        -----------------
        arguments:
        df - pd.dataframe / Pandas dataframe containing the information for training
        augment - Bool/ whether you want to apply data augmentation
        batch_size - None,int / Batch-size for ds. If none specified -> take the one from utils.py
        seed - to generate same train/val split when reloading model
        val_split - Split apart a small ds for accuracy estimations
        return_eval_data - whether to return data for evaluation (test_ds + df)
        -----------------
        returns:
        Training and Validation Dataset.
        """

        if batch_size is None:
            batch_size = util.BATCH_SIZE  # if no batch size specified, we take the one from utils.py
            print(f"Since none Batch-size was specified we, took the {batch_size} specified in utils.py")

        # Create order for the batches
        train_df, val_df = smart_batches(df, batch_size, task="individual", seed=seed, val_split=val_split)

        train_ds = self.build_ds(train_df["image"], train_df["label"])
        if augment:
            train_ds = train_ds.map(self.augment)
        train_ds = train_ds.batch(batch_size).prefetch(10)

        val_ds = self.build_ds(val_df["image"], val_df["label"])
        val_ds = val_ds.batch(batch_size).prefetch(10)

        if not return_eval_data:
            return train_ds
        elif return_eval_data:
            return train_ds, val_ds, train_df, val_df

    def build_ds(self, image_paths, labels):
        """
        Build the final dataset.
        -----------------
        arguments:
        image_paths - filepath to resized images
        labels - (species) labels
        -----------------
        returns:
        Dataset
        """

        image_paths = util.TRAIN_DATA_PATH + "/" + image_paths
        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(self.prepare_images_mapping, num_parallel_calls=8)

        return ds

    def generate_single_individuals_ds(self, df, batch_size):
        """
        Build dataset which also contains individuals with only one image.
        -----------------
        arguments:
        df - pandas.DataFrame with all images
        batch_size - batch size
        -----------------
        returns:
        Dataset and dedicated pandas.DataFrame.
        """

        df = df[df["individual_counts"] == 1]

        return self.return_plain_ds(df, batch_size), df

    def return_plain_ds(self, df, batch_size):
        """
        Build dataset.
        -----------------
        arguments:
        df - pandas.DataFrame with all images
        batch_size - batch size
        -----------------
        returns:
        Dataset
        """
        return self.build_ds(df["image"], df["label"]).batch(batch_size).prefetch(10)