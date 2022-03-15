import os
import pandas as pd
import numpy as np
from PIL import Image
from util import IMG_CSV, IMG_FOLDER

class Extractor():
    
    def __init__(self):
        pass
    
    def extract_individuals(self, n_most_common=100):
        
        most_common = IMG_CSV.value_counts(subset=["individual_id"])[:n_most_common]
        most_common_indices = most_common[:n_most_common].index.values
        ids = [i for t in most_common_indices for i in t]
        imgs = IMG_CSV[IMG_CSV.loc[:,"individual_id"].isin(ids)].reset_index(drop=True)
        
        imgs.to_csv("imgs_subset")
        np.save("most_common_indices", most_common_indices)
        
    def extract_shapes(self):
        
        img_shapes = []

        for i in IMG_CSV.loc[:,"image"]:
            shape = Image.open(os.path.join(IMG_FOLDER, i)).size
            img_shapes.append(shape)
            
        img_shapes = np.array(img_shapes)
        np.save("img_shapes", img_shapes)