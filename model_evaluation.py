import numpy as np
from scipy.spatial.distance import cdist
import tensorflow_datasets as tfds

def mean_average_precision(model, val_ds):
    # get the embeddings
    embedding_results = model.predict(val_ds)
    # get the corresponding_labels
    corresponding_labels = np.concatenate([label for img, label in tfds.as_numpy(val_ds)])
    assert len(embedding_results) == len(corresponding_labels)

    # get pairwise distance matrix
    pairwise_distances = cdist(embedding_results, embedding_results)

    # array for index of closest datapoint
    closest_data_indeces = np.zeros(len(pairwise_distances), dtype=int)

    # for every datapoint
    for i in range(len(pairwise_distances)):
        # get distance column
        arr = pairwise_distances[i]
        # remove distance to itself
        arr = np.concatenate((arr[:i], arr[i + 1:]))
        # get closest index
        closest_data_indeces[i] = np.argmin(arr)

    # get predicted labels
    predicted_labels = np.array([corresponding_labels[i] for i in closest_data_indeces])

    return np.mean(predicted_labels == corresponding_labels)