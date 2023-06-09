import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data
    n_samples = 750
    n_features = 2
    random_state = 40

    x_train = generate_random_data(n_samples, n_features, random_state)

    # Fit the One-Class SVM model
    nu = 0.1  # Contamination level
    clf = OneClassSVM(nu=nu)
    clf.fit(x_train)

    # Predict anomaly scores
    scores = clf.decision_function(x_train)

    # Plot the data points with their anomaly scores
    plt.scatter(x_train[:, 0], x_train[:, 1], c=-scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('One-Class SVM Anomaly Detection')
    plt.show()


if __name__ == '__main__':
    main()
