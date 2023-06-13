import random

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Generate random data
    n_samples = 1000
    n_features = 2
    n_classes = 2
    random_state = random.randint(0, 100)

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state
    )

    # Fit the KNN model
    k = 3
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y)

    # Predict class labels for all data points
    y_pred = clf.predict(x)

    # Plot the data points with their predicted labels
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, cmap='viridis')
    plt.colorbar(label='Predicted Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KNN Classification (k={k})')
    plt.show()

    return y_pred


if __name__ == '__main__':
    main()
