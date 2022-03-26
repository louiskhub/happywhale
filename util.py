"""
Python script containing all global variables and filepaths.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import pandas as pd
import numpy as np

# Target Shape
TARGET_SHAPE = (256,256)
# Folder in which the Original Kaggle Training Images reside
IMG_FOLDER = "../KaggleData/train_images"

IMG_CSV = pd.read_csv("../KaggleData/train.csv")

UPPER_LIMIT_OF_IMAGES = 10

give_nice_percentage = lambda x,y: int( 100 * np.round(x/y,2))

PATH_FOR_OUR_TRAINING_DATA = "../OurSpeciesTrainingData"

BATCH_SIZE = 64

training_df = pd.read_csv(PATH_FOR_OUR_TRAINING_DATA + "/data.csv", index_col=0)