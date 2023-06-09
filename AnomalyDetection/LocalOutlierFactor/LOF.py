import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data with outliers
    n_samples = 500
    n_features = 2
    random_state = 40
    x_train = generate_random_data(n_samples, n_features, random_state)

    # Fit the LOF model
    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(x_train)

    # Plot the training data with outlier labels
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_pred, cmap='viridis')
    plt.colorbar(label='Outlier Label')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Local Outlier Factor')
    plt.show()


if __name__ == '__main__':
    main()
