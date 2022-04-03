"""
Crop all the images to a certain size to decrease the training data drastically.
@author: fhammer, lkapp
"""

import tensorflow_datasets as tfds
import util
import tqdm
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="dataset_creation")
parser.add_argument(
    "-w", "--write_dir", help="path to the directory you want to save your images in",
    default=util.TRAIN_DATA_PATH
)
args = parser.parse_args()


df = util.TRAIN_SPECIES_DF
ds = tf.data.Dataset.from_tensor_slices(df["image"])


def mapping_func(path):
    """
    Function to be mapped on tfds to read+downsize images.
    """
    img = tf.io.read_file( "../KaggleData/train_images/" + path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, util.TARGET_SHAPE)
    return (img, path)


ds = ds.map(mapping_func)

# save every image to the path
for img, path in tqdm.tqdm(tfds.as_numpy(ds)):
    tf.keras.utils.save_img(args.write_dir + path.decode("utf-8"), img)