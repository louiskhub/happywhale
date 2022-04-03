"""
All functions evaluation of triplet loss models.
@authors: fhammer, lkapp
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.visualizer import nicer_classes
import plotnine as p9
import pandas as pd
import time
from scipy.spatial.distance import cdist
import umap
from matplotlib.patches import Patch


def compute_closest_k_neighbors(a, b, k=3):
    pairwise_dis = cdist(a, b)
    vals, indeces = tf.nn.top_k(tf.math.negative(pairwise_dis), k)
    vals = tf.math.negative(vals)
    return vals, indeces


def k_accuracy(train_df, val_df, k, closest_k_indices, indivums=True, return_mean=True):
    assert closest_k_indices.shape[1] >= k

    if indivums:
        loc_column = "label"
        iloc_column = 4
    else:
        loc_column = "species_label"
        iloc_column = 5

    real_labels = np.array(val_df[loc_column])
    closest_k_indices = closest_k_indices[:, :k]

    labels = np.concatenate([np.expand_dims(train_df.iloc[closest_k_indices[:, i], iloc_column], -1) for i in range(k)],
                            axis=1)
    hits = np.isin(real_labels, labels)

    if return_mean:
        return np.mean(hits)
    else:
        return hits


def calculate_class_accuracy(species, val_df):
    df = val_df.loc[
        val_df["species"] == species, ["correctly_labeled", "species_correctly_labeled", "correctly_labeled_k5"]]
    acc = np.mean(df["correctly_labeled"])
    spec_acc = np.mean(df["species_correctly_labeled"])
    k5_acc = np.mean(df["correctly_labeled_k5"])
    return np.array([acc, spec_acc, k5_acc])


def create_triplet_eval(model, train_ds, val_ds, train_df, val_df, new_whales_ds, new_whales_df, path_to_save, name):
    # calculate embeddings

    print("Calculate Embeddings")
    start_time = time.time()

    val_embeddings = model.predict(val_ds, verbose=1)
    train_embeddings = model.predict(train_ds, verbose=1)
    new_whales_embeddings = model.predict(new_whales_ds, verbose=1)

    # compute closest neighbors

    print(f"Finished, took {start_time - time.time()}s. Now compute closest neighbors")
    start_time = time.time()

    val_closest_distances, val_closest_k_indices = compute_closest_k_neighbors(val_embeddings, train_embeddings, 5)
    new_whales_closest_distances, _ = compute_closest_k_neighbors(new_whales_embeddings, train_embeddings, 1)

    # compute embeddings

    print(f"Finished, took {start_time - time.time()}s. Now compute umap embeddings")
    start_time = time.time()
    reducer = umap.UMAP()
    reduced_train_embeddings = reducer.fit_transform(train_embeddings)
    reduced_val_embeddings = reducer.fit_transform(val_embeddings)

    print(f"Finished, took {start_time - time.time()}s. Now create plots")

    # compute accuracies
    val_df["correctly_labeled"] = k_accuracy(train_df, val_df, 1, val_closest_k_indices, indivums=True,
                                             return_mean=False)
    val_df["species_correctly_labeled"] = k_accuracy(train_df, val_df, 1, val_closest_k_indices, indivums=False,
                                                     return_mean=False)
    val_df["correctly_labeled_k5"] = k_accuracy(train_df, val_df, 5, val_closest_k_indices, indivums=True,
                                                return_mean=False)



    over_all_acc = np.round(np.mean(val_df["correctly_labeled"]), 4)
    over_all_acc_species = np.round(np.mean(val_df["species_correctly_labeled"]), 4)
    over_all_top5_acc = np.round(np.mean(val_df["correctly_labeled_k5"]), 4)

    fig_info = name + f"Accuracies:  Individual-MAP: {over_all_acc*100}%, Species-MAP: {over_all_acc_species*100}%, k5-MAP - {over_all_top5_acc*100}%"

    fig, ax = plt.subplots(figsize=(14, 14))

    for species in train_df["species"].unique():
        mask = (train_df["species"] == species).values
        ax.scatter(reduced_train_embeddings[mask, 0], reduced_train_embeddings[mask, 1], label=nicer_classes(species))

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.figtext(0.5, 0.08, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.title("2d UMAP Visualisation of Train Embeddings", fontsize=18)

    plt.savefig(path_to_save + "2dUMAP_train.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(14, 14))

    for species in val_df["species"].unique():
        mask = (val_df["species"] == species).values
        ax.scatter(reduced_val_embeddings[mask, 0], reduced_val_embeddings[mask, 1], label=nicer_classes(species))

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.figtext(0.5, 0.08, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.title("2d UMAP Visualisation of Val Embeddings", fontsize=18)
    plt.savefig(path_to_save + "2dUMAP_val.png", bbox_inches='tight')

    # create daframe for density plot
    old_distances = np.expand_dims(val_closest_distances[:, 0], -1).tolist()
    old_distances = [i + ["Old Whales"] for i in old_distances]
    new_distances = [i + ["New Whales"] for i in new_whales_closest_distances.numpy().tolist()]
    df = pd.DataFrame(new_distances + old_distances, columns=['Distance', 'Kind'])

    mean_old = np.round(np.mean(val_closest_distances[:, 0]),2)
    mean_new = np.round(np.mean(old_distances[:, 0]),2)
    p = (p9.ggplot(data=df,
                   mapping=p9.aes(x='Distance', color='Kind'))
         + p9.geom_density()
         + p9.ggtitle(f"Distance to next neighbor,means: old: {mean_old}, new: {mean_new}"))
    p.save(filename=path_to_save + "distances_to_next_whale_density.png", height=5, width=5, units='in', dpi=1000)

    names = [nicer_classes(n) for n in list(val_df["species"].unique())]
    data = [calculate_class_accuracy(species, val_df) for species in names]
    acc = [data[i][0] for i in range(len(data))]
    spec_acc = [data[i][1] for i in range(len(data))]
    top5_acc = [data[i][2] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(names, spec_acc, color='#1f77b4', alpha=0.5)
    ax.barh(names, top5_acc, color='#ff7f0e', alpha=0.7)
    ax.barh(names, acc, color='#d62728', alpha=0.8)
    legend_elements = [Patch(facecolor='#1f77b4', label='Species-MAP'),
                       Patch(facecolor='#ff7f0e', label='k5-MAP'),
                       Patch(facecolor='#d62728', label='Individual-MAP')]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Kind of Accuracy")
    plt.title("Accuracy by Class", fontsize=18)
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)
    plt.figtext(0.5, 0.08, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(path_to_save + "accuracy_by_class.png", bbox_inches='tight')

    val_df.to_csv(path_to_save + "val_df.csv")
