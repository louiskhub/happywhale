import pandas as pd
import numpy as np

TARGET_SHAPE = (300, 300)
IMG_FOLDER = "../train_images/"
IMG_CSV_SUBSET = pd.read_csv("ckpts/imgs_subset.csv").dropna().reset_index(drop=True)
IMG_CSV = pd.read_csv("ckpts/imgs.csv").dropna().reset_index(drop=True)
IMG_SHAPES = np.load("ckpts/img_shapes.npy", allow_pickle=True)
MOST_COMMON_INDIVIDUALS = np.load("ckpts/most_common_indices.npy", allow_pickle=True)
