import json

from AnomalyDetection.IsolationForest.IF import main as IFmain
from AnomalyDetection.IsolationNearestNeighborsEnsemble.INNE import main as INNEmain
from AnomalyDetection.LocalOutlierFactor.LOF import main as LOFmain
from AnomalyDetection.OneClassSVM.OCSvm import main as OCSVMmain
from AnomalyDetection.Other.COPOD import main as COPODmain
from AnomalyDetection.Other.GMM import main as GMMmain
from AnomalyDetection.utilsAD import generate_random_data


def main():
    # Generate random data
    x_train = generate_random_data(750, 2, 56)

    plot = True

    # Calculate scores for different algorithms
    scores = {
        'Isolation Forest': IFmain(x_train, plot),
        'Local Outlier Factor': LOFmain(x_train, plot),
        'One-Class SVM': OCSVMmain(x_train, plot),
        'Copula-based Outlier Detection': COPODmain(x_train, plot),
        'Gaussian Mixture Models': GMMmain(x_train, plot),
        'Isolation Nearest Neighbors Ensemble': INNEmain(x_train, plot)
    }

    # Combine x, y coordinates with algorithm scores
    data = []
    for i in range(len(x_train)):
        point = {
            'x': x_train[i][0],
            'y': x_train[i][1]
        }
        point.update({f"{algo_name} score": score[i] for algo_name, score in scores.items()})
        data.append(point)

    # Export data to JSON
    with open('AnomalyScore.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)


if __name__ == '__main__':
    main()
