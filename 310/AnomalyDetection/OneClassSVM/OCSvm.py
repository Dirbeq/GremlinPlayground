from sklearn.svm import OneClassSVM

from AnomalyDetection.utilsAD import generate_random_data, plot_data


def main(x_train, plot=False):
    # Fit the One-Class SVM model
    nu = 0.1  # Contamination level
    clf = OneClassSVM(nu=nu)
    clf.fit(x_train)

    # Predict anomaly scores
    scores = clf.decision_function(x_train)

    if plot:
        # Plot the data points with their anomaly scores
        plot_data(x_train, -scores, 'One-Class SVM (OCSVM)')

    return -scores


if __name__ == '__main__':
    main(generate_random_data(750, 2, 40), plot=True)
