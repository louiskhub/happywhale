"""
Script to define all functions for Visualizations.

Louis Kapp, Felix Hammer, Yannik Ullrich
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from util import  TRAIN_IMG_FOLDER

# nice plot funcs

def nicer_classes(name):
    name = name.replace("_", " ")
    name = name[0].capitalize() + name[1:]
    i = name.find(" ")
    return name[:i + 1] + name[i + 1].capitalize() + name[i + 2:]

def insert_missing_class(new_dis):
    global org_dis
    for species in org_dis:
        if species not in new_dis:
            new_dis[species]=0
    return new_dis

def get_class_distribution(df, count_barrier=None):
    counts = df["species"].apply(nicer_classes).value_counts(normalize=True)
    if count_barrier is not None:
        counts = counts[counts < count_barrier]
    return {name: val for name, val in zip(counts.index, counts.values)}


def insert_missing_class(new_dis,org_dis):
    for species in org_dis:
        if species not in new_dis:
            print
            new_dis[species] = 0
    return new_dis

def get_distributions(org_df, train_df, val_df, cut_of_val=None):
    org_dis, train_dis, val_dis = [get_class_distribution(x,cut_of_val) for x in [org_df, train_df, val_df]]
    train_dis, val_dis = [insert_missing_class(x,org_dis) for x in  [train_dis, val_dis]]

    return org_dis, train_dis,val_dis
def plot_class_bars(org_df, train_df, val_df, cut_of_val=None):
    org_dis, train_dis,val_dis = get_distributions(org_df, train_df, val_df, cut_of_val=None)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.bar(org_dis.keys(), org_dis.values(), label="Original class distribution", fill=False, edgecolor='green')
    ax.bar(val_dis.keys(), val_dis.values(), label="Validation class distribution", fill=False, edgecolor='blue')
    ax.bar(train_dis.keys(), train_dis.values(), label="Train class distribution", fill=False, edgecolor='red')
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.show()


def plot_class_barh(org_df, train_df, val_df, cut_of_val=None):
    org_dis, train_dis,val_dis = get_distributions(org_df, train_df, val_df, cut_of_val=None)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(list(org_dis.keys()), org_dis.values(), label="Original class distribution", fill=False, edgecolor='green')
    ax.barh(list(val_dis.keys()), val_dis.values(), label="Validation class distribution", fill=False, edgecolor='blue')
    ax.barh(list(train_dis.keys()), train_dis.values(), label="Train class distribution", fill=False, edgecolor='red')
    plt.legend()
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)
    plt.show()


def plot_relative_diffs(rel_train_difs, rel_val_difs, rel_test_difs,whale_index):

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.barh(whale_index, rel_train_difs, label="Relative Training Proportions", fill=False,
            edgecolor='orange')
    ax.barh(whale_index, rel_val_difs, label="Relative Training Proportions", fill=False,
            edgecolor='yellowgreen')
    ax.barh(whale_index, rel_test_difs, label="Relative Val Proportions", fill=False, edgecolor='cyan')

    plt.legend()
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)
    plt.show()


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

    for i, img in enumerate(PRELIM_TRAIN_DF.loc[:9, "image"]):
        image = plt.imread(os.path.join(TRAIN_IMG_FOLDER, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i // 5, i % 5].set_title(PRELIM_TRAIN_DF.iloc[i, 2])
        ax[i // 5, i % 5].imshow(image)

    plt.tight_layout()
    plt.show()
