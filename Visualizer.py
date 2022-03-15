import matplotlib.pyplot as plt
from DS_Generator import DS_Generator
import cv2
import os
from util import IMG_CSV, IMG_FOLDER, TARGET_SHAPE

class Visualizer():
    
    def __init__(self, ds=None, individuals_path=None):
        
        if ds == None:
            if individuals_path == None:
                self.ds = DS_Generator().preprocess()
            else:
                self.ds = DS_Generator().preprocess(individuals_path)
        else:
            self.ds = ds
        
    def plot_preprocessed(self):
        
        def plot(anchor, positive, negative):
            def show(ax, image):
                ax.imshow(image)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            fig, ax = plt.subplots(3, 3)
            
            for i in range(3):
                show(ax[i, 0], anchor[:,:,i])
                show(ax[i, 1], positive[:,:,i])
                show(ax[i, 2], negative[:,:,i])
            
            plt.tight_layout()
            plt.show()
        
        plot(*list(self.ds.take(1).as_numpy_iterator())[0])
        
    def plot_original(self):
        
        fig, ax = plt.subplots(2, 5, figsize=(40,20))
        
        for i, img in enumerate(IMG_CSV.loc[:9, "image"]):
            image = plt.imread(os.path.join(IMG_FOLDER, img))
            resized = cv2.resize(image, (TARGET_SHAPE,TARGET_SHAPE))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ax[i//5, i%5].set_title(IMG_CSV.iloc[i, 2])
            ax[i//5, i%5].imshow(image)
            
        plt.tight_layout()
        plt.show()