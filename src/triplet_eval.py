"""
All functions evaluation of triplet loss models.
@authors: fhammer, lkapp
"""

import os

if os.getcwd()[-10:] != "happywhale":
    os.chdir("..")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src import visualizer
import plotnine as p9
import pandas as pd
import time
from scipy.spatial.distance import cdist
import umap
from matplotlib.patches import Patch


def compute_closest_k_neighbors(a, b, k=3):
    """
    This function computes the k closest neighbors of the embeddings and their respective values
    ----------------
    arguments:
    a - (M,d) Matrix
    b - (N,d) Matrix
    k - amount of neighbors
    -----------------
    returns:
    vals - (M,k) of distance to k nearest neighbors, sorted
    indices - (M,k) of indeces of k nearest neighbors in b
    """
    # get paiwise distance
    pairwise_dis = cdist(a, b)
    #
    vals, indeces = tf.nn.top_k(tf.math.negative(pairwise_dis), k)
    vals = tf.math.negative(vals)
    return vals, indeces


def k_accuracy(train_df, val_df, k, closest_k_indices, individuals=True, return_mean=True):
    """
    This function computes the k-MAP for different both the species and the individuals
    ----------------
    arguments:
    train_df - your training data frane
    val_df - your validation data frame
    k - setting for MAP
    closest_k_indices - the indices of the closest neighbors for the train_df
    individuals - True -> Compute individual MAP, False compute Species MAP
    return_mean - whether you want to return the meaned values or the columns
    -----------------
    returns:
    np.mean(hits) - MAP value if return_mean == True
    hits - raw column vector, with the individual hits
    """
    assert closest_k_indices.shape[1] >= k

    if individuals:
        loc_column = "label"
        iloc_column = train_df.columns.tolist().index("label")
    else:
        loc_column = "species_label"
        iloc_column = train_df.columns.tolist().index("species_label")

    real_labels = np.array(val_df[loc_column])
    closest_k_indices = closest_k_indices[:, :k]

    labels = np.concatenate([np.expand_dims(train_df.iloc[closest_k_indices[:, i], iloc_column], -1) for i in range(k)],
                            axis=1)
    hits = np.array([np.isin(ar1, ar2) for ar1, ar2 in zip(real_labels, labels)])

    if return_mean:
        return np.mean(hits)
    else:
        return hits


def calculate_class_accuracy(species, val_df):
    """
    calculated the accuracies for a specific class
    :param species: string of the species in the df
    :param val_df: df with accs column already in it
    :return: MAP,Species-MAP , k5-MAP
    """
    df = val_df.loc[
        val_df["species"] == species, ["correctly_labeled", "species_correctly_labeled", "correctly_labeled_k5"]]
    acc = np.mean(df["correctly_labeled"])
    spec_acc = np.mean(df["species_correctly_labeled"])
    k5_acc = np.mean(df["correctly_labeled_k5"])
    return np.array([acc, spec_acc, k5_acc])


def create_triplet_eval(model, train_ds, val_ds, train_df, val_df, new_whales_ds, path_to_save, name):
    """
    This functions is there to evaluate our siamese triplet loss models
    :param model: the model you want evaluate
    :param train_ds: the data set your model has been training on, order must still be the same as the train_df
    :param val_ds: the dataset you want to use for evaluation
    :param train_df: the dataframe with the train_ds info
    :param val_df: the dataframe with the val_ds infos
    :param new_whales_ds: the dataset of the animals which have only one image, are new to the training dataset
    :param path_to_save: the path were you want to save the plots
    :param name: The Name of the model at this stage, f.e.: Control Model - Epoch 05
    :return: None
    """
    print("Calculate Embeddings")
    start_time = time.time()

    val_embeddings = model.predict(val_ds, verbose=1)
    train_embeddings = model.predict(train_ds, verbose=1)
    new_whales_embeddings = model.predict(new_whales_ds, verbose=1)

    print(f"Finished, took {np.round(time.time() - start_time)}s. Now compute closest neighbors")
    start_time = time.time()

    # compute closest neighbors for both the validation dataset and the new whales with the train embeddings

    val_closest_distances, val_closest_k_indices = compute_closest_k_neighbors(val_embeddings, train_embeddings, 5)
    new_whales_closest_distances, _ = compute_closest_k_neighbors(new_whales_embeddings, train_embeddings, 1)

    # compute embeddings

    print(f"Finished, took {np.round(time.time() - start_time)}s. Now compute UMAP embeddings")
    start_time = time.time()
    reducer = umap.UMAP()
    reduced_train_embeddings = reducer.fit_transform(train_embeddings)
    reduced_val_embeddings = reducer.fit_transform(val_embeddings)

    print(f"Finished, took {np.round(time.time() - start_time)}s. Now create plots")

    # compute accuracies
    val_df["correctly_labeled"] = k_accuracy(train_df, val_df, 1, val_closest_k_indices, individuals=True,
                                             return_mean=False)
    val_df["species_correctly_labeled"] = k_accuracy(train_df, val_df, 1, val_closest_k_indices, individuals=False,
                                                     return_mean=False)
    val_df["correctly_labeled_k5"] = k_accuracy(train_df, val_df, 5, val_closest_k_indices, individuals=True,
                                                return_mean=False)

    over_all_acc = np.round(np.mean(val_df["correctly_labeled"]), 4)
    over_all_acc_species = np.round(np.mean(val_df["species_correctly_labeled"]), 4)
    over_all_top5_acc = np.round(np.mean(val_df["correctly_labeled_k5"]), 4)

    fig_info = name + f"Accuracies:  Individual-MAP: {over_all_acc * 100}%, Species-MAP: {over_all_acc_species * 100}%, k5-MAP: {over_all_top5_acc * 100}%"

    fig, ax = plt.subplots(figsize=(14, 14))

    for species in train_df["species"].unique():
        mask = (train_df["species"] == species).values
        ax.scatter(reduced_train_embeddings[mask, 0], reduced_train_embeddings[mask, 1], label=visualizer.nicer_species_names(species))

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.figtext(0.5, 0.08, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.title("2d UMAP Visualisation of Train Embeddings", fontsize=18)

    plt.savefig(path_to_save + "2dUMAP_train.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(14, 14))

    for species in val_df["species"].unique():
        mask = (val_df["species"] == species).values
        ax.scatter(reduced_val_embeddings[mask, 0], reduced_val_embeddings[mask, 1], label=visualizer.nicer_species_names(species))

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.figtext(0.5, 0.08, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.title("2d UMAP Visualisation of Val Embeddings", fontsize=18)
    plt.savefig(path_to_save + "2dUMAP_val.png", bbox_inches='tight')

    # create daframe for density plot
    old_distances = np.expand_dims(val_closest_distances[:, 0], -1).tolist()
    old_distances = [i + ["Old Whales"] for i in old_distances]
    new_distances = [i + ["New Whales"] for i in new_whales_closest_distances.numpy().tolist()]
    df = pd.DataFrame(new_distances + old_distances, columns=['Distance', 'Kind'])

    mean_old = np.round(np.mean(val_closest_distances[:, 0]), 2)
    mean_new = np.round(np.mean(new_whales_closest_distances[:, 0]), 2)
    p = (p9.ggplot(data=df,
                   mapping=p9.aes(x='Distance', color='Kind'))
         + p9.geom_density()
         + p9.ggtitle(f"Distance to next neighbor,means: old: {mean_old}, new: {mean_new}"))
    p.save(filename=path_to_save + "distances_to_next_whale_density.png", height=5, width=5, units='in', dpi=1000)

    names = [n for n in list(val_df["species"].unique())]
    data = [calculate_class_accuracy(species, val_df) for species in names]
    acc = [data[i][0] for i in range(len(data))]
    spec_acc = [data[i][1] for i in range(len(data))]
    top5_acc = [data[i][2] for i in range(len(data))]
    names = [visualizer.nicer_species_names(n) for n in names]
    fig, ax = plt.subplots(figsize=(14, 14))
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
    plt.figtext(0.5, 0.02, fig_info, wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(path_to_save + "accuracy_by_class.png", bbox_inches='tight')

    val_df.to_csv(path_to_save + "val_df.csv")


