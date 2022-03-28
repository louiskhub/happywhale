"""
Python script containing all global variables

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import pandas as pd


################################################################
# LOCAL FILEPATHS ##############################################
################################################################

# original data from https://www.kaggle.com/competitions/whale-categorization-playground/data
TRAIN_IMG_FOLDER = "../../KaggleData/train_images"
TRAIN_CSV = "../../KaggleData/train.csv"

# local path to our modified version of the kaggle data
TRAIN_DATA_PATH = "../../OurTrainingData"


################################################################
# PREPROCESSING ################################################
################################################################

TARGET_SHAPE = (224, 224) # because of imagenet
UPPER_LIMIT_OF_IMAGES = 10  # max. 10 images per individual to reduce overfitting

PRELIM_TRAIN_DF = pd.read_csv(TRAIN_CSV)
#TRAIN_INDIVIDUAL_DF = pd.read_csv(TRAIN_DATA_PATH + "/individual_data.csv", index_col=0)
#TRAIN_SPECIES_DF = pd.read_csv(TRAIN_DATA_PATH + "/species_data.csv", index_col=0)

################################################################
# TRAINING #####################################################
################################################################

BATCH_SIZE = 64
