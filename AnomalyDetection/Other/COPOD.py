import matplotlib.pyplot as plt
from pyod.models.copod import COPOD

from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data
    n_samples = 750
    n_features = 2
    random_state = 40

    x_train = generate_random_data(n_samples, n_features, random_state)

    # Fit the COPOD model
    clf = COPOD()
    clf.fit(x_train)

    # Predict outlier scores
    scores = clf.decision_function(x_train)

    # Plot the data points with their outlier scores
    plt.scatter(x_train[:, 0], x_train[:, 1], c=-scores, cmap='viridis')
    plt.colorbar(label='Outlier Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Copula-Based Outlier Detection (COPOD)')
    plt.show()

    return scores


if __name__ == '__main__':
    main()
