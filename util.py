"""
Python script containing all global variables

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import pandas as pd


################################################################
# LOCAL FILEPATHS ##############################################
################################################################

# original data from https://www.kaggle.com/competitions/whale-categorization-playground/data
IMG_FOLDER = "../KaggleData/train_images"
IMG_CSV = pd.read_csv("../KaggleData/train.csv")

# local path to our modified version of the kaggle data
TRAIN_DATA_PATH = "../OurSpeciesTrainingData"


################################################################
# PREPROCESSING ################################################
################################################################

TARGET_SHAPE = (256, 256)
UPPER_LIMIT_OF_IMAGES = 10  # max. 10 images per individual to reduce overfitting
TRAIN_DF = pd.read_csv(TRAIN_DATA_PATH + "/data.csv", index_col=0)


################################################################
# TRAINING #####################################################
################################################################

BATCH_SIZE = 64
