from gremlin_python.structure.graph import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


def generate_data(g):
    # Generate data
    g.addV('Person').property('name', 'Alice').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Bob').property('age', 27).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Charlie').property('age', 32).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Diana').property('age', 39).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Eve').property('age', 20).property('nb_client', 4).next()
    g.addV('Person').property('name', 'Frank').property('age', 80).property('nb_client', 30).next()
    g.addV('Person').property('name', 'Grace').property('age', 2).property('nb_client', 40).next()
    g.addV('Person').property('name', 'Helen').property('age', 25).property('nb_client', 9).next()
    g.addV('Person').property('name', 'Ivan').property('age', 33).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Julia').property('age', 28).property('nb_client', 3).next()
    g.addV('Person').property('name', 'Kevin').property('age', 31).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Linda').property('age', 40).property('nb_client', 25).next()
    g.addV('Person').property('name', 'Michael').property('age', 35).property('nb_client', 7).next()
    g.addV('Person').property('name', 'Nancy').property('age', 32).property('nb_client', 8).next()
    g.addV('Person').property('name', 'Olivia').property('age', 30).property('nb_client', 5).next()
    g.addV('Person').property('name', 'Peter').property('age', 32).property('nb_client', 4).next()
    g.addV('Person').property('name', 'Quentin').property('age', 27).property('nb_client', 6).next()
    g.addV('Person').property('name', 'Rachel').property('age', 30).property('nb_client', 7).next()


# Connect to the graph database
graph = Graph()
connection = DriverRemoteConnection('ws://localhost:8182/gremlin', 'g')
g = graph.traversal().withRemote(connection)

# Clear all data from the graph database
g.V().drop().iterate()

generate_data(g)
# Retrieve all vertices from the graph database
vertices = g.V().hasLabel('Person').valueMap(True).toList()

# Extract features for LOF analysis
X = [[v['nb_client'][0], v['age'][0]] for v in vertices]

# Perform LOF analysis
n_neighbors = min(10, len(vertices) - 1)  # Adjust the value based on your needs
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
y_pred = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_

# Print LOF scores next to each data point
for i, v in enumerate(vertices):
    print(f"{v['name'][0]}: {scores[i]:.2f}")

# Plot
plt.title("Local Outlier Factor (LOF)")
plt.scatter([x[0] for x in X], [x[1] for x in X], color="k", s=3.0, label="Data points")
plt.scatter([x[0] for i, x in enumerate(X) if y_pred[i] == -1], [x[1] for i, x in enumerate(X) if y_pred[i] == -1],
            color="r", s=30.0, label="Outliers")
for i, v in enumerate(vertices):
    if y_pred[i] == -1:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]), color="r")
    else:
        plt.annotate(v['name'][0], (X[i][0], X[i][1]))

radius = (scores.max() - scores) / (scores.max() - scores.min())
plt.xlabel("Number of clients")
plt.ylabel("Age")
plt.legend(loc="upper left")
plt.show()
# Close the connection to the graph database
connection.close()
