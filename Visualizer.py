"""
Script to define all functions for Visualizations.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import matplotlib.pyplot as plt
import cv2
import os
from util import IMG_CSV, IMG_FOLDER


def plot_preprocessed(anchor, positive, negative):
    """
    Shows an example of a Triplet.
    
    Parameters
    ----------
    anchor
    positive
    negative
    """
    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig, ax = plt.subplots(3, 3)

    for i in range(3):
        show(ax[i, 0], anchor[:, :, i])
        show(ax[i, 1], positive[:, :, i])
        show(ax[i, 2], negative[:, :, i])

    plt.tight_layout()
    plt.show()


def plot_original():
    """
    
    """
    fig, ax = plt.subplots(2, 5, figsize=(40, 20))

    for i, img in enumerate(IMG_CSV.loc[:9, "image"]):
        image = plt.imread(os.path.join(IMG_FOLDER, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i // 5, i % 5].set_title(IMG_CSV.iloc[i, 2])
        ax[i // 5, i % 5].imshow(image)

    plt.tight_layout()
    plt.show()
