"""
Extracts the Images with the most common individuals and the image shapes.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import os
import numpy as np
from PIL import Image
from util import IMG_CSV, IMG_FOLDER
import pandas as pd
from tqdm import tqdm


def create_our_training_data(IMG_FOLDER, TRAIN_DATA_PATH, TARGET_SHAPE):
    df = pd.read_csv(TRAIN_DATA_PATH + "/data.csv", index_col=0)
    
    image_paths = df["image"].values
    
    for i in tqdm(range(len(image_paths))):
        read_path = IMG_FOLDER + "/" + image_paths[i]
        img = Image.open(read_path)

        img = img.resize(TARGET_SHAPE)
        
        write_path = TRAIN_DATA_PATH + "/" + image_paths[i]
        img.save(write_path)


def extract_shapes():
    """
    Extracts the shapes of the images from the IMG_CSV and saves them.
    """
    img_shapes = []

    for i in IMG_CSV.loc[:, "image"].dropna():
        shape = Image.open(os.path.join(IMG_FOLDER, str(i))).size
        img_shapes.append(shape)

    img_shapes = np.array(img_shapes)
    np.save("ckpts/img_shapes", img_shapes)
