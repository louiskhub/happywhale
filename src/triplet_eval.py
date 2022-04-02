import sys

sys.path.append("..")
sys.path.append("../src")
import numpy as np
import matplotlib.pyplot as plt
import umap
from src.model_evaluation import compute_closest_k_neighbors
from src.Visualizer import nicer_classes
import plotnine as p9
import pandas as pd
import time
from matplotlib.patches import Patch

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

    labels = np.concatenate([np.expand_dims(train_df.iloc[closest_k_indices[:, i], iloc_column], -1) for i in range(k)], axis=1)

    hits = np.array([np.isin(col1, col2) for col1,col2 in zip(real_labels, labels)])

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


def create_triplet_eval(model, train_ds, val_ds, train_df, val_df, new_whales_ds, new_whales_df, path_to_save,name):
    start = time.time()
    print("Start computing the embeddings")
    val_preds = model.predict(val_ds, verbose=1)
    train_preds = model.predict(train_ds, verbose=1)
    new_whales_preds = model.predict(new_whales_ds, verbose=1)

    # small helper
    round = lambda x: np.round(x, 4)

    print(f"Took {round(time.time()-start)}s, now compute closest neighbors")
    start = time.time()
    val_closest_vals, closest_k_indices = compute_closest_k_neighbors(val_preds, train_preds, 5)
    new_whales_closest_vals, _ = compute_closest_k_neighbors(new_whales_preds, train_preds, 1)

    print(f"Took {round(time.time()-start)}s, now the umap embeddings")
    start = time.time()
    val_embeddings = umap.UMAP().fit_transform(val_preds)
    train_embeddings = umap.UMAP().fit_transform(train_preds)
    print(f"Took {round(time.time()-start)}s, now the plots")


    # Compute means
    val_df["correctly_labeled"] = k_accuracy(train_df, val_df, 1, closest_k_indices, indivums=True, return_mean=False)
    val_df["species_correctly_labeled"] = k_accuracy(train_df, val_df, 1, closest_k_indices, indivums=False, return_mean=False)
    val_df["correctly_labeled_k5"] = k_accuracy(train_df, val_df, 5, closest_k_indices, indivums=True, return_mean=False)
    over_all_acc = round(np.mean(val_df["correctly_labeled"])) * 100
    over_all_acc_species = round(np.mean(val_df["species_correctly_labeled"])) * 100
    over_all_top5_acc = round(np.mean(val_df["correctly_labeled_k5"])) * 100

    summary_str = f"Model: {name}, MAP: {over_all_acc}%, K5-MAP: {over_all_top5_acc}%, Species-MAP: {over_all_acc_species}%"
    print(summary_str)

    # Plot UMAP train embeddings

    fig, ax = plt.subplots(figsize=(14, 14))

    for species in train_df["species"].unique(): # for every species

        mask = (train_df["species"] == species).values # get the respective indexes
        # plot embeddings and color by class
        ax.scatter(train_embeddings[mask, 0], train_embeddings[mask, 1], label=nicer_classes(species))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.title(f"2d UMAP Visualisation of Train Embedding Space", fontsize=20)
    plt.figtext(0.5, 0.08, summary_str, wrap=True, horizontalalignment='center', fontsize=16)
    plt.savefig(path_to_save + "2dUMAP.png", bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(14, 14))
    for species in val_df["species"].unique():
        mask = (val_df["species"] == species).values
        ax.scatter(val_embeddings[mask, 0], val_embeddings[mask, 1], label=nicer_classes(species))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Species")
    plt.title(f"2d UMAP Visualisation of Validation Embedding Space", fontsize=20)
    plt.figtext(0.5, 0.08, summary_str, wrap=True, horizontalalignment='center', fontsize=16)
    plt.savefig(path_to_save + "3dUMAP.png", bbox_inches='tight')

    # create df containing the distance to the next whale for a density plot with plotnine
    old_distances = np.expand_dims(val_closest_vals[:, 0], -1).tolist()
    old_distances = [i + ["Old Whales"] for i in old_distances]
    new_distances = [i + ["New Whales"] for i in new_whales_closest_vals.numpy().tolist()]
    df = pd.DataFrame(new_distances + old_distances, columns=['Distance', 'Kind'])
    # compute the means
    mean_old = round(np.mean(val_closest_vals[:, 0]))
    mean_new = round(np.mean(new_whales_closest_vals[:, 0]))
    # plot with plotnine
    p = (p9.ggplot(data=df,
                   mapping=p9.aes(x='Distance', color='Kind'))
         + p9.geom_density()
         + p9.ggtitle(f"Distance to next neighbor,means: old: {np.round(mean_old,2)}, new: {np.round(mean_new,2)}"))
    p.save(filename=path_to_save + "distances_to_next_whale_density.png", height=5, width=5, units='in', dpi=500)

    # Get the Accuracies sorted by classs
    names = [n for n in list(val_df["species"].unique())]
    data = [calculate_class_accuracy(species, val_df) for species in names]
    acc = [data[i][0] for i in range(len(data))]
    spec_acc = [data[i][1] for i in range(len(data))]
    top5_acc = [data[i][2] for i in range(len(data))]

    # create nicer names for plotting
    names = [nicer_classes(n) for n in names]

    # Plot accuracy by class
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.barh(names, spec_acc, color='#1f77b4', alpha=0.5)
    ax.barh(names, top5_acc, color='#ff7f0e', alpha=0.7)
    ax.barh(names, acc, color='#d62728', alpha=0.8)
    plt.title("Accuracy by class", fontsize=20)

    plt.figtext(0.5, 0.065, summary_str, wrap=True, horizontalalignment='center', fontsize=16)

    legend_elements = [Patch(facecolor='#d62728', label='Individual-MAP'),
                       Patch(facecolor='#ff7f0e', label='k5-MAP'),
                       Patch(facecolor='#1f77b4', label='Species-MAP')]

    ax.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1, 0.5), title="Accuracies by Class")
    plt.xticks(rotation=0)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.ylabel('Whale Species', fontsize=16)
    plt.xlabel('Counts', fontsize=16)

    plt.savefig(path_to_save + "accuracy_by_class.png", bbox_inches='tight')
    # save val df
    val_df.to_csv(path_to_save + "val_df.csv")

    return over_all_acc, over_all_top5_acc, over_all_acc_species

