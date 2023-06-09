import numpy as np
from sklearn.datasets import make_blobs


def generate_random_data(n_samples, n_features, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    global_random_state = 12  # Change this to a specific value for all algorithms, or None for random
    if global_random_state is not None:
        np.random.seed(global_random_state)

    centers = [[-4, -4], [4, 4], [8, 8]]
    x_train, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)

    # Generate outliers
    n_outliers = int(n_samples * 0.05)  # % of samples as outliers
    x_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, n_features))
    x_train = np.concatenate((x_train, x_outliers), axis=0)

    return x_train
