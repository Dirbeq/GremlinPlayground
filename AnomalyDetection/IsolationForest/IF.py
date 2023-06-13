from sklearn.ensemble import IsolationForest

from AnomalyDetection.utilsAD import generate_random_data, plot_data


def main(x_train, plot=False):
    # Fit the Isolation Forest model
    clf = IsolationForest()
    clf.fit(x_train)

    # Predict anomaly scores for training data
    scores_train = clf.decision_function(x_train)

    if plot:
        # Plot the data points with their anomaly scores
        plot_data(x_train, -scores_train, 'Isolation Forest (IF)')

    return -scores_train


if __name__ == '__main__':
    main(generate_random_data(750, 2, 40), plot=True)
