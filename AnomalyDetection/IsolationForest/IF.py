import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from AnomalyDetection.utilsAD import generate_random_data

# Generate random data with outliers
n_samples = 1000
n_features = 2
random_state = 0
X_train = generate_random_data(n_samples, n_features, random_state=random_state)

# Fit the Isolation Forest model
clf = IsolationForest(random_state=random_state)
clf.fit(X_train)

# Predict anomaly scores for training data
scores_train = clf.decision_function(X_train)

# Plot the training data with anomaly scores
plt.scatter(X_train[:, 0], X_train[:, 1], c=-scores_train, cmap='viridis')
plt.colorbar(label='Anomaly Score')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Isolation Forest Anomaly Detection')
plt.show()
