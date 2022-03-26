"""
Creates a Tensorflow dataset and Triplets for the Triplet loss function.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from util import PATH_FOR_OUR_TRAINING_DATA, UPPER_LIMIT_OF_IMAGES, BATCH_SIZE, training_df
import math
import random

def df_filter_for_indidum_training(train_df:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    ids, respective_counts = np.unique(train_df["individual_id"].values,return_counts=True)
    ids = ids[respective_counts>1] # boolean index the ids 

    respective_counts= respective_counts[respective_counts>1] # and counts for later

    filter_function = np.vectorize(lambda x: x in ids) # our filter function

    train_df = train_df.iloc[filter_function(train_df["individual_id"])] # filter df

    train_df.index = range(len(train_df)) # reindex
    return train_df



def smart_batches(df:pd.core.frame.DataFrame, BATCH_SIZE:int, task:str = "individual") -> pd.core.frame.DataFrame:
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
    assert task in ["individual","species"], 'task has to be either "individual_id" or "species"" and must be column index of df'
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
    
   
    assert BATCH_SIZE%2==0, "BATCH_SIZE must be even"
    

    df["assign_to"] = np.nan

    even_mask = (df[counts_column]%2==0).array
    uneven_mask = np.logical_not(even_mask)

    even_indices_list = list(df[even_mask].index)
    uneven_df = df[uneven_mask]

    amount_of_containers = math.ceil(len(df)/BATCH_SIZE)
    container = np.array([BATCH_SIZE for i in range(amount_of_containers-1) ] + [len(df)%BATCH_SIZE])

    set_of_uneven_classes = {a for a in uneven_df[label]}
    
    
    if not len(set_of_uneven_classes)%2 != container[-1]%2:
        unlucky_class = random.choice(uneven_df.index)
        df.drop(index = unlucky_class )

        even_mask = (df[counts_column]%2==0).array
        uneven_mask = np.logical_not(even_mask)

        even_indices_list = list(df[even_mask].index)
        uneven_df = df[uneven_mask]
        set_of_uneven_classes = {a for a in uneven_df[label]}
        
        print(f"We threw away the datapoint with index {unlucky_class} ")
    
    uneven_labels = {a:[] for a in uneven_df[label].array}
    for index, int_label in zip(uneven_df.index,  uneven_df[label].array):
        uneven_labels[int_label].append(index)
        
    for int_label in uneven_labels:
        if len(uneven_labels[int_label])>3:
            rest = uneven_labels[int_label][3:]
            keep = uneven_labels[int_label][:3]
            even_indices_list.extend(rest)

            uneven_labels[int_label] = keep
            
    uneven_indices_list = [uneven_labels[a] for a in uneven_labels]
    random.shuffle(uneven_indices_list)
    
    if len(set_of_uneven_classes)%2==1:
        container[-1]-=3
        first_triplet = uneven_indices_list.pop()
        df.loc[first_triplet,"assign_to"]=len(container)-1
    assert len(uneven_indices_list)%2 == 0, "stf went horbly wrong"

    combined_double_triplets = [a+b for a,b in zip(uneven_indices_list[::2],uneven_indices_list[1::2])]
    assert all([len(a) == 6 for a in combined_double_triplets])
    
    even_df = df.loc[even_indices_list]
    even_labels = even_df[label].sort_values().index
    
    combined_even_doubles = [[a,b] for a,b in zip(even_labels[::2],even_labels[1::2])]
    random.shuffle(combined_even_doubles)

    assert all([df.loc[a,label]==df.loc[b,label] for a,b in combined_even_doubles])
    i = 0
    while combined_double_triplets:


        if container[i]<6:
            i = i+1 if i+1!=len(container) else 0
            continue

        triplets = combined_double_triplets.pop()
        container[i]-=6
        df.loc[triplets,"assign_to"]=i

        i = i+1 if i+1!=len(container) else 0 

    i = 0
    while combined_even_doubles:


        if container[i]<2:
            i = i+1 if i+1!=len(container) else 0
            continue

        double = combined_even_doubles.pop()
        container[i]-=2
        df.loc[double,"assign_to"]=i

        i = i+1 if i+1!=len(container) else 0 


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
