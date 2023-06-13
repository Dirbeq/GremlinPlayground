from pyod.models.lof import LOF

from AnomalyDetection.utilsAD import generate_random_data, plot_data


def main(x_train, plot=False):
    # Fit the LOF model
    clf = LOF()
    clf.fit(x_train)

    # Predict outlier scores
    scores = clf.decision_function(x_train)

    if plot:
        # Plot the data points with their outlier scores
        plot_data(x_train, scores, 'Local Outlier Factor (LOF)')

    return scores


if __name__ == '__main__':
    main(generate_random_data(750, 2, 40), plot=True)
