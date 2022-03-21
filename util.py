"""
Python script containing all global variables and filepaths.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import pandas as pd
import numpy as np

# Target Shape
TARGET_SHAPE = (300, 300)
# Folder in which the Training Images reside
IMG_FOLDER = "../train_images/"
# CSV with subset of images (most common individuals)
IMG_CSV_SUBSET = pd.read_csv("ckpts/imgs_subset.csv").dropna().reset_index(drop=True)
# 
IMG_CSV = pd.read_csv("ckpts/imgs.csv").dropna().reset_index(drop=True)
#
IMG_SHAPES = np.load("ckpts/img_shapes.npy", allow_pickle=True)
# 
MOST_COMMON_INDIVIDUALS = np.load("ckpts/most_common_indices.npy", allow_pickle=True)
