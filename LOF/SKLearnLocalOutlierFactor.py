from gremlin_python.structure.graph import Graph
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

from sklearn.neighbors import LocalOutlierFactor

# Connect to the graph database
graph = Graph()
connection = DriverRemoteConnection('ws://localhost:8182/gremlin', 'g')
g = graph.traversal().withRemote(connection)

# Clear all data from the graph database
g.V().drop().iterate()

# Generate data
g.addV('Person').property('name', 'Alice').property('age', 30).next()
g.addV('Person').property('name', 'Bob').property('age', 27).next()
g.addV('Person').property('name', 'Charlie').property('age', 32).next()
g.addV('Person').property('name', 'Diana').property('age', 70).next()
g.addV('Person').property('name', 'Eve').property('age', 20).next()
g.addV('Person').property('name', 'Frank').property('age', 29).next()
g.addV('Person').property('name', 'Grace').property('age', 34).next()
g.addV('Person').property('name', 'Helen').property('age', 25).next()
g.addV('Person').property('name', 'Ivan').property('age', 33).next()
g.addV('Person').property('name', 'Julia').property('age', 28).next()
g.addV('Person').property('name', 'Kevin').property('age', 31).next()
g.addV('Person').property('name', 'Linda').property('age', 26).next()
g.addV('Person').property('name', 'Michael').property('age', 35).next()

# Retrieve all vertices from the graph database
vertices = g.V().hasLabel('Person').valueMap(True).toList()

# Extract features for LOF analysis (example: use 'age' property)
X = [[v['age'][0]] for v in vertices]

# Perform LOF analysis
n_neighbors = min(10, len(vertices) - 1)  # Adjust the value based on your needs
lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
y_pred = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_

# Print LOF scores next to each data point
for i, v in enumerate(vertices):
    print(f"{v['name'][0]}: {scores[i]:.2f}")


# Close the connection to the graph database
connection.close()
