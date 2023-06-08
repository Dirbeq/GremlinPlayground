from gremlin_python.structure.graph import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from matplotlib import pyplot as plt
from numpy import random
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def generate_data(g):
    # Generate data
    g.addV('Person').property('name', 'Alice').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Bob').property('age', 27).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Charlie').property('age', 32).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Diana').property('age', 39).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Eve').property('age', 20).property('nb_client', 4).next()
    g.addV('Person').property('name', 'Frank').property('age', 80).property('nb_client', 30).next()
    g.addV('Person').property('name', 'Grace').property('age', 2).property('nb_client', -20).next()
    g.addV('Person').property('name', 'Helen').property('age', 25).property('nb_client', 9).next()
    g.addV('Person').property('name', 'Ivan').property('age', 33).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Julia').property('age', 28).property('nb_client', 3).next()
    g.addV('Person').property('name', 'Kevin').property('age', 31).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Linda').property('age', 40).property('nb_client', 35).next()
    g.addV('Person').property('name', 'Michael').property('age', 35).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Nancy').property('age', 32).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Olivia').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Peter').property('age', 32).property('nb_client', 4).next()
    g.addV('Person').property('name', 'Quentin').property('age', 27).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Rachel').property('age', 30).property('nb_client', 7).next()
    # generate ppl randomly but with a concentration on age 30 and nb_client 7
    for i in range(10):
        g.addV('Person').property('name', 'Random' + str(i)).property('age', 30 + random.randint(-10, 10)).property(
            'nb_client', 7 + random.randint(-10, 10)).next()

    for i in range(10):
        g.addV('Person').property('name', 'Random' + str(i)).property('age', 140 + random.randint(-10, 10)).property(
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
n_neighbors = min(10, len(vertices) - 1)  # Adjust the value based on your needs
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
y_pred = lof.fit_predict(X)
lof_scores = -lof.negative_outlier_factor_

# Perform Isolation Forest analysis
n_estimators = 100  # Adjust the value based on your needs
isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=0.1)
isolation_forest.fit(X)
if_scores = isolation_forest.decision_function(X)

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
for i, v in enumerate(vertices):
    if y_pred[i] == -1:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="r")
    elif if_scores[i] < 0:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="b")
    else:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]))

plt.xlabel("Number of clients")
plt.ylabel("Age")
plt.legend(loc="upper left")
plt.show()

# Close the connection to the graph database
connection.close()
