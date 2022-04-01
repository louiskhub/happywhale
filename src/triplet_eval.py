import sys
sys.path.append("..")
sys.path.append("../src")
from src import DS_Generator
import tensorflow as tf
import matplotlib.pyplot as plt
from util import TRAIN_SPECIES_DF,TRAIN_DATA_PATH, INDIVIDUMS_SEED
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.model_evaluation import compute_closest_k_neighbors
import tensorflow_datasets as tfds
import seaborn as sns
from src.Visualizer import nicer_classes
import plotnine as p9
import pandas as pd


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

def calculate_class_accuracy(species,val_df):
    df = val_df.loc[val_df["species"]==species,["correctly_labeled","species_correctly_labeled","correctly_labeled_k5"]]
    acc = np.mean(df["correctly_labeled"])
    spec_acc =  np.mean(df["species_correctly_labeled"])
    k5_acc = np.mean(df["correctly_labeled_k5"])
    return np.array([acc, spec_acc,k5_acc])


def create_triplet_eval(model,train_ds,val_ds,train_df,val_df,new_whales_ds,new_whales_df,path_to_save):

    val_preds = model.predict(val_ds, verbose=1)
    train_preds = model.predict(train_ds, verbose=1)
    new_whales_preds = model.predict(new_whales_ds, verbose=1)

    vals, closest_k_indices = compute_closest_k_neighbors(val_preds, train_preds, 5)

    val_df["correctly_labeled"] = k_accuracy(train_df, val_df, 1, closest_k_indices, indivums=True, return_mean=False)
    val_df["species_correctly_labeled"] = k_accuracy(train_df, val_df, 1, closest_k_indices, indivums=False,
                                                     return_mean=False)
    val_df["correctly_labeled_k5"] = k_accuracy(train_df, val_df, 5, closest_k_indices, indivums=True,
                                                return_mean=False)

    f = lambda x: np.round(x, 2)
    over_all_acc = f(np.mean(val_df["correctly_labeled"]))
    over_all_acc_species = f(np.mean(val_df["species_correctly_labeled"]))
    over_all_top5_acc = f(np.mean(val_df["correctly_labeled_k5"]))

    sns.set_theme()
    transform_embeddings_2d = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(train_preds)
    fig, ax = plt.subplots(figsize=(14, 14))


    for species in train_df["species"].unique():
        mask = (train_df["species"] == species).values
        ax.scatter(transform_embeddings_2d[mask, 0], transform_embeddings_2d[mask, 1], label=nicer_classes(species))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.title(
        f"2d Visualisation of Embedding Space - Individum MAP: {over_all_acc}, Species MAP: {over_all_acc_species}, Top 5 MAP - {over_all_top5_acc}")
    plt.savefig(path_to_save + "2dTSNE.png",bbox_inches='tight')

    transform_embeddings_3d = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(train_preds)
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')

    for species in train_df["species"].unique():
        mask = (train_df["species"] == species).values
        ax.scatter(transform_embeddings_3d[mask, 0], transform_embeddings_3d[mask, 1], transform_embeddings_3d[mask, 2],
                   label=nicer_classes(species))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.title(
        f"3d Visualisation of Embedding Space - Individum MAP: {over_all_acc}, Species MAP: {over_all_acc_species}, Top 5 MAP - {over_all_top5_acc}")
    plt.savefig(path_to_save + "3dTSNE.png",bbox_inches='tight')

    new_vals, _ = compute_closest_k_neighbors(new_whales_preds, train_preds, 1)

    old_distances = np.expand_dims(vals[:, 0], -1).tolist()
    old_distances = [i + ["Old Whales"] for i in old_distances]
    new_distances = [i + ["New Whales"] for i in new_vals.numpy().tolist()]
    df = pd.DataFrame(new_distances + old_distances, columns=['Distance', 'Kind'])

    mean_old = f(np.mean(vals[:, 0]))
    mean_new = f(np.mean(new_vals[:, 0]))
    p = (p9.ggplot(data=df,
                   mapping=p9.aes(x='Distance', color='Kind'))
         + p9.geom_density()
         + p9.ggtitle(f"Distance to next neighbor,means: old: {mean_old}, new: {mean_new}"))
    p.save(filename=path_to_save + "distances_to_next_whale_density.png", height=5, width=5, units='in', dpi=1000)

    names = list(val_df["species"].unique())
    data = [calculate_class_accuracy(species, val_df) for species in names]
    acc = [data[i][0] for i in range(len(data))]
    spec_acc = [data[i][1] for i in range(len(data))]
    top5_acc = [data[i][2] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(names, spec_acc, color='#1f77b4', alpha=0.5)
    ax.barh(names, top5_acc, color='#ff7f0e', alpha=0.7)
    ax.barh(names, acc, color='#d62728', alpha=0.8)
    plt.title("Accuracy by class, Red=Indivum_MAP, Orange=K5_Individum_MAP, Blue=Species_MAP")
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)

    plt.savefig(path_to_save + "accuracy_by_class.png",bbox_inches='tight')

    val_df.to_csv(path_to_save + "val_df.csv")

