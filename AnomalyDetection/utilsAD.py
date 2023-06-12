import json

import numpy as np
from sklearn.datasets import make_blobs


def generate_random_data(n_samples, n_features, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    global_random_state = 42  # Change this to a specific value for all algorithms, or None for random
    if global_random_state is not None:
        np.random.seed(global_random_state)

    centers = [[0, -4], [4, 6], [8, 7]]
    x_train, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)

    # Generate outliers
    n_outliers = int(n_samples * 0.05)  # % of samples as outliers
    x_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, n_features))
    x_train = np.concatenate((x_train, x_outliers), axis=0)

    return x_train


def export_to_json(x_train, scores, filename):
    scores = np.array([scores]).transpose().tolist()
    data = np.concatenate((x_train, scores), axis=1)

    data = data.tolist()
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
