"""
Crop all the images to a certain size to decrease the training data drastically.
@authors: fhammer, lkapp
"""

import tensorflow_datasets as tfds
import util
import tqdm
import tensorflow as tf
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="dataset_creation")
parser.add_argument(
    "-w", "--write_dir", help="path to the directory you want to save your images in",
    default=util.TRAIN_DATA_PATH
)
args = parser.parse_args()


df = util.TRAIN_DF
ds = tf.data.Dataset.from_tensor_slices(df["image"])


def mapping_func(image_path):
    """
    Function to be mapped on tfds to read+downsize images.
    -----------------
    arguments:
    image_path - filename of specific image
    -----------------
    returns:
    Resized image and its filename.
    """

    image = tf.io.read_file("../KaggleData/train_images/" + image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, util.TARGET_SHAPE)
    return image, image_path


ds = ds.map(mapping_func)

# save every image to the path
for img, path in tqdm.tqdm(tfds.as_numpy(ds)):
    tf.keras.utils.save_img(args.write_dir + "/" + path.decode("utf-8"), img)


# add df columns for training

# create columns with counts
df["species_counts"] = df.groupby('species')["species"].transform('count')
df['individual_counts'] = df.groupby('individual_id')['individual_id'].transform('count')

# create int labels for species/individuals
int_labels = {name: label for label, name in enumerate(df["individual_id"].unique())}
df["label"] = df["individual_id"].apply(lambda x: int_labels[x])

species_labels = {name: label for label, name in enumerate(df["species"].unique())}
df["species_label"] = df["species"].apply(lambda x: species_labels[x])

individual_counts = df["individual_counts"].values
species_counts = df["species_counts"].values

pd.to_csv(args.write_dir + "/" + "train_data.csv")
