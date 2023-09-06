import csv

from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.structure.graph import Graph
from matplotlib import pyplot as plt
from numpy import random
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm


def generate_data(g):
    # Generate data
    g.addV('Person').property('name', 'Alice').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Bob').property('age', 27).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Charlie').property('age', -32).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Diana').property('age', 39).property('nb_client', -8).next()
    g.addV('Person').property('name', 'Eve').property('age', 20).property('nb_client', 4).next()
    g.addV('Person').property('name', 'Frank').property('age', 80).property('nb_client', 30).next()
    g.addV('Person').property('name', 'Grace').property('age', 2).property('nb_client', -20).next()
    g.addV('Person').property('name', 'Helen').property('age', 25).property('nb_client', 9).next()
    g.addV('Person').property('name', 'Ivan').property('age', 33).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Julia').property('age', -28).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Kevin').property('age', 31).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Linda').property('age', 40).property('nb_client', 60).next()
    g.addV('Person').property('name', 'Michael').property('age', 35).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Nancy').property('age', -35).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Olivia').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Peter').property('age', -32).property('nb_client', 50).next()
    g.addV('Person').property('name', 'Quentin').property('age', -30).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Rachel').property('age', 120).property('nb_client', 7).next()
    # generate ppl randomly but with a concentration on age 30 and nb_client 7
    for i in tqdm(range(20)):
        g.addV('Person').property('name', 'Random1.' + str(i)).property('age', 30 + random.randint(-10, 10)).property(
            'nb_client', 7 + random.randint(-10, 10)).next()

    for i in tqdm(range(20)):
        g.addV('Person').property('name', 'Random2.' + str(i)).property('age', 140 + random.randint(-10, 10)).property(
            'nb_client', 50 + random.randint(-10, 10)).next()


# Connect to the graph database
graph = Graph()
connection = DriverRemoteConnection('ws://localhost:8182/gremlin', 'g')
g = graph.traversal().withRemote(connection)

# Clear all data from the graph database
g.V().drop().iterate()

generate_data(g)
# Retrieve all vertices from the graph database
vertices = g.V().hasLabel('Person').valueMap(True).toList()

# Extract features for LOF and Isolation Forest analysis
X = [[v['nb_client'][0], v['age'][0]] for v in vertices]

# Perform LOF analysis
n_neighbors = min(15, len(vertices) - 1)  # Adjust the value based on your needs
contamination = 0.12  # Adjust the value based on your needs
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

# Perform Isolation Forest analysis
n_estimators = 115  # Adjust the value based on your needs
isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination)
isolation_forest.fit(X)
if_scores = isolation_forest.decision_function(X)

# Create the dataset with age, nb_client, and is_outlier from data.csv
dataset = []
fileFound = True
try:
    with open('data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            dataset.append([int(row[0]), int(row[1]), int(row[2])])

    # Fit a Random Forest classifier
    Classifier = RandomForestClassifier(n_estimators=100)
    X_train = [[x[0], x[1]] for x in dataset]
    y_train = [x[2] for x in dataset]
    Classifier.fit(X_train, y_train)

    # Generate predictions for new data points
    new_data = []
    # random data
    for i in range(520):
        new_data.append([random.randint(-25, 80), random.randint(-25, 150)])
    predictions = Classifier.predict(new_data)

    # Print the predictions
    for i, data_point in enumerate(new_data):
        print(f"Data point {i + 1}: Age={data_point[0]}, nb_client={data_point[1]}, is_outlier={predictions[i]}")
    print("-" * 100)
except FileNotFoundError:
    fileFound = False
    print("File not found")

# Append IF data to existing data.csv
with open('data.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i, v in enumerate(vertices):
        age = v['age'][0]
        nb_client = v['nb_client'][0]
        is_outlier = 1 if (if_scores[i] < 0) else 0
        writer.writerow([age, nb_client, is_outlier])

# Calculate Local Outlier Probability (LOP)
lof_probas = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

# Calculate Isolation Forest Local Outlier Probability (IF-LOP)
if_probas = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

# Print LOF scores, LOP, and Isolation Forest scores, IF-LOP next to each data point
for i, v in enumerate(vertices):
    print(
        f"{v['name'][0]}: LOF={lof_scores[i]:.2f}, LOP={lof_probas[i]:.2f}, IF={if_scores[i]:.2f}, IF-LOP={if_probas[i]:.2f}")

# Plot
plt.title("Outlier Detection")
plt.scatter([x[0] for x in X], [x[1] for x in X], color="k", s=3.0, label="Data points")
plt.scatter([x[0] for i, x in enumerate(X) if y_pred[i] == -1], [x[1] for i, x in enumerate(X) if y_pred[i] == -1],
            color="r", s=30.0, label="LOF Outliers")
plt.scatter([x[0] for i, x in enumerate(X) if if_scores[i] < 0], [x[1] for i, x in enumerate(X) if if_scores[i] < 0],
            color="b", s=30.0, label="IF Outliers")
plt.scatter([x[0] for i, x in enumerate(X) if y_pred[i] == -1 and if_scores[i] < 0],
            [x[1] for i, x in enumerate(X) if y_pred[i] == -1 and if_scores[i] < 0], color="m", s=30.0,
            label="LOF & IF Outliers")
if fileFound:
    plt.scatter([x[0] for i, x in enumerate(new_data) if predictions[i] > 0.0],
                [x[1] for i, x in enumerate(new_data) if predictions[i] > 0.0], color="g", s=30.0,
                label="Random Forest Outliers")
    plt.scatter([x[0] for i, x in enumerate(new_data) if predictions[i] == 0.0],
                [x[1] for i, x in enumerate(new_data) if predictions[i] == 0.0], color="y", s=30.0,
                label="Random Forest inlines")

for i, v in enumerate(vertices):
    if y_pred[i] == -1 and if_scores[i] < 0:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="m")
    else:
        if y_pred[i] == -1:
            plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="r")
        elif if_scores[i] < 0:
            plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="b")
        else:
            # plt.annotate(v['name'][0], (X[i][0], X[i][1]))
            pass

plt.xlabel("Number of clients")
plt.ylabel("Age")
plt.legend(loc="upper left")
plt.show()

# Close the connection to the graph database
connection.close()
