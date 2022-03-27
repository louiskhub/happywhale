import numpy as np
from scipy.spatial.distance import cdist
import tensorflow_datasets as tfds

def mean_average_precision(model,train_ds , val_ds):
    embeddings = model.predict(train_ds)

    corresponding_labels = np.concatenate([label for img, label in tfds.as_numpy(train_ds)])

    val_embeddings = model.predict(val_ds)
    val_labels = np.concatenate([label for img, label in tfds.as_numpy(val_ds)])

    pairwise_distances = cdist(embeddings, val_embeddings)

    closest_embeddings_indices = np.argmin(pairwise_distances, 0)
    predicted_labels = np.array([corresponding_labels[i] for i in closest_embeddings_indices])

    return np.mean(predicted_labels == val_labels)
