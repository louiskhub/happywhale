"""
All functions for Visualizations.
@authors: fhammer, lkapp
"""

import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
import util
import numpy as np
import cv2


def nicer_species_names(name):
    """
    Reformat a species name for plotting.
    -----------------
    arguments:
    name - String of species name to modify
    -----------------
    returns:
    Modified String of species name.
    """

    name = name.replace("_", " ")
    name = name[0].capitalize() + name[1:]
    i = name.find(" ")

    return name[:i + 1] + name[i + 1].capitalize() + name[i + 2:]


def get_species_distribution(df, count_barrier=None):
    """
    Reformat a species name for plotting.
    -----------------
    arguments:
    name - String of species name to modify
    -----------------
    returns:
    Modified String of species name.
    """

    counts = df["species"].apply(nicer_species_names).value_counts(normalize=True)
    if count_barrier is not None:
        counts = counts[counts < count_barrier]

    return {name: val for name, val in zip(counts.index, counts.values)}


def plot_original(prelim_df):
    fig, ax = plt.subplots(2, 5, figsize=(40, 20))

    for i, img in enumerate(prelim_df.loc[:9, "image"]):
        image = plt.imread(os.path.join(util.TRAIN_IMG_FOLDER, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax[i // 5, i % 5].set_title(prelim_df.iloc[i, 2])
        ax[i // 5, i % 5].imshow(image)

    plt.tight_layout()
    plt.show()


def eval_soft_max_plot_test_whales(test_ds, best_3_indices, best_3_vals, columns=3, rows=40):
    fig = plt.figure(figsize=(20, 160))
    for i, (img, label, name) in enumerate(tfds.as_numpy(test_ds.take(columns * rows))):
        # for every row/column
        # check wether result was corect or in top 3
        if label == best_3_indices[i, 0]:
            certainty = str(int(100 * np.round(best_3_vals[i, 0], 2)))
            result = f"Correct with {certainty}%"
        elif label in best_3_indices[i]:
            pos = np.argwhere(label == best_3_indices[i])[0][0]
            result = f"False, {pos + 1} position"

        else:
            result = "Not in best 3"

        # plot images
        img = img
        labels = labels
        img += 1
        img /= 2
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
        name = name.decode("utf-8")
        plt.title(f"{name} - {result}")
        plt.axis("off")
    plt.show()


def eval_soft_max_plot_acc_by_class(test_df, best_3_indices, best_3_vals):
    # calculate the accs by class
    class_acc, class_top3_acc, class_freq = {}, {}, {}
    for species in test_df["species"].unique():
        name = nicer_species_names(species)
        df = test_df[test_df["species"] == species]
        species_label = df["species_label"].iloc[0]
        class_acc[name] = np.round(np.mean(best_3_indices[df.index.tolist(), 0] == species_label), 2)
        class_top3_acc[name] = np.round(np.mean([species_label in arr for arr in best_3_indices[df.index.tolist()]]), 2)
        class_freq[name] = np.round(len(df) / len(test_df), 2)

    # plot them
    names = list(class_acc.keys())
    fig = plt.figure(figsize=(14, 14))
    plt.barh(names, class_top3_acc.values())
    plt.barh(names, class_acc.values())
    plt.barh(names, class_freq.values())
    plt.title("Accuracy by class")
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)
    plt.show()
