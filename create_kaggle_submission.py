"""
Create the submission.csv for the kaggle Happywhale competition.
@authors: fhammer, lkapp
"""

from src import ds_generator, models
import tensorflow as tf
import util
import os
import tqdm
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="submission_creation")
parser.add_argument(
    "-w", "--write_dir", help="path to the directory you want to save your submission csv in",
    default=util.SUBMISSION_PATH
)
parser.add_argument(
    "-c", "--control_model", help="get results from the control model",
    default=False
)
parser.add_argument(
    "-n", "--new_model", help="get results from a new model",
    default=False
)
parser.add_argument(
    "-p", "--pretrained_model", help="get results from the pretrained model",
    default=True
)
parser.add_argument(
    "--pretrained_weighs", help="filepath to pretrained weights (use in combination with -p or -c)",
    default=util.PRETRAINED_WEIGHTS
)
parser.add_argument(
    "-d", "--distance_new_whales", help="default distance for new whales",
    default=0.35
)
args = parser.parse_args()

test_df = pd.DataFrame(os.listdir(util.TEST_DATA_PATH), columns=["image"])
test_df["label"] = 0
test_ds = ds_generator.DS_Generator().return_plain_ds(test_df, 64, train_imgs=False)
whole_train_ds = ds_generator.DS_Generator().return_plain_ds(util.TRAIN_DF, 64)


# some model you want to try
if args.control_model:
    model = models.return_siamese_control_model(args.pretrained_weighs)
elif args.new_model:
    model = models.return_new_siamese_model()
elif args.pretrained_model:
    model = models.return_soft_max_pretrained_siamese_model(args.pretrained_weighs)

train_embedding = model.predict(whole_train_ds, verbose=1)
test_embedding = model.predict(test_ds, verbose=1) 


# save just in case
np.save(args.write_dir + "/train_embedding.npy", train_embedding)
np.save(args.write_dir + "/test_embedding.npy", test_embedding )


# compute the pairwise_distance (this will take long)
pairwise_distance = cdist(test_embedding, train_embedding)  # distance_‘cosine’
# and also save it just in case
np.save(args.write_dir + "/pairwise_distance.npy", pairwise_distance)


# split it up to not overstress memory and then calculate best 5 vals/indexes
split_ups = np.ceil(np.linspace(0, pairwise_distance.shape[0])).astype(int)
closest_vals, closest_indices = list(), list()
for i in tqdm.tqdm(range(1, len(split_ups))):
    split = pairwise_distance[split_ups[i-1]:split_ups[i]]
    vals, indexes = tf.nn.top_k(tf.math.negative(split), 5)
    vals = tf.math.negative(vals)
    closest_vals.append(vals)
    closest_indices.append(indexes)


# re-concatenate
best_vals = np.concatenate(closest_vals)
best_indexes = np.concatenate(closest_indices)


for i in tqdm.tqdm(range(len(test_df))):
    distances = best_vals[i]
    labels = [util.TRAIN_DF.iloc[k, 2] for k in best_indexes[i]]

    # if the distance to a neighboring whale is bigger then the distance for new whales, insert a new whale
    args = np.argwhere(distances > args.distance_new_whales)
    if len(args) != 0:
        first = int(args[0, 0])
        labels.insert(first, "new_individual")

    test_df.iloc[i, -1] = " ".join(labels[:5])


# write submission.csv
test_df.to_csv(args.write_dir + "/submission.csv", index=False, columns=["image", "predictions"])
