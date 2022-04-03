"""
Python script containing all global variables

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import pandas as pd

################################################################
# LOCAL FILEPATHS ##############################################
################################################################

# local path to our modified version of the kaggle data
TRAIN_DATA_PATH = "../OurTrainingData"
SAVING_PATH = "../models/"
TEST_DATA_PATH = "../OurTestData"
SUBMISSION_PATH = "submission"

################################################################
# PREPROCESSING ################################################
################################################################

TARGET_SHAPE = (224, 224) # because of imagenet
TRAIN_SPECIES_DF = pd.read_csv(TRAIN_DATA_PATH + "/species_data.csv", index_col=0)

################################################################
# TRAINING #####################################################
################################################################

SPECIES_SEED = 5
INDIVIDUALS_SEED = 0
NUMBER_OF_SPECIES = 30
BATCH_SIZE = 64

