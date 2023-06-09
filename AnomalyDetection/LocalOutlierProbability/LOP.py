import matplotlib.pyplot as plt
from pyod.models.lof import LOF

from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data
    n_samples = 500
    n_features = 2
    random_state = 40

    x = generate_random_data(n_samples, n_features, random_state)

    # Fit the LOP model
    contamination = 0.1  # Percentage of outliers
    clf = LOF(contamination=contamination)
    clf.fit(x)

    # Predict outlier scores
    scores = clf.decision_function(x)

    # Plot the data points with their outlier scores
    plt.scatter(x[:, 0], x[:, 1], c=-scores, cmap='viridis')
    plt.colorbar(label='Outlier Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Local Outlier Probability (LOP)')
    plt.show()


if __name__ == '__main__':
    main()
