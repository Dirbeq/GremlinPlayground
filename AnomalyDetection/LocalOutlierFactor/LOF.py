import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

from AnomalyDetection.utilsAD import generate_random_data

# Generate random data with outliers
n_samples = 1000
n_features = 2
random_state = 42
X_train = generate_random_data(n_samples, n_features, random_state=random_state)

# Fit the LOF model
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(X_train)

# Plot the training data with outlier labels
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap='viridis')
plt.colorbar(label='Outlier Label')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Local Outlier Factor')
plt.show()
