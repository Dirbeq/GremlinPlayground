import numpy as np
from sklearn.datasets import make_blobs


def generate_random_data(n_samples, n_features, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    centers = [[-2, -2], [2, 2]]
    x_train, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)

    # Generate outliers
    n_outliers = int(n_samples * 0.1)  # 10% of samples as outliers
    x_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, n_features))
    x_train = np.concatenate((x_train, x_outliers), axis=0)

    return x_train
