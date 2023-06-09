import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data with outliers
    n_samples = 500
    n_features = 2
    random_state = 40
    x_train = generate_random_data(n_samples, n_features, random_state)

    # Fit the Isolation Forest model
    clf = IsolationForest(random_state=random_state)
    clf.fit(x_train)

    # Predict anomaly scores for training data
    scores_train = clf.decision_function(x_train)

    # Plot the training data with anomaly scores
    plt.scatter(x_train[:, 0], x_train[:, 1], c=-scores_train, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Isolation Forest Anomaly Detection')
    plt.show()


if __name__ == '__main__':
    main()
