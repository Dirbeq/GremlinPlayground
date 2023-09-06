import json

import matplotlib.pyplot as plt

# Load data
with open('INNE.json') as f:
    data = json.load(f)

# Extract x, y, and score values
x = [row['x'] for row in data]
y = [row['y'] for row in data]
scores = [row['score'] for row in data]

# Create scatter plot
plt.scatter(x, y, c=scores, cmap='plasma')
plt.colorbar(label='Score')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()

sensitivity = 0.9

# Only show the outliers scores > 0.9
x2 = [row['x'] for row in data if row['score'] >= sensitivity]
y2 = [row['y'] for row in data if row['score'] >= sensitivity]
scores2 = [row['score'] for row in data if row['score'] >= sensitivity]

# Create scatter plot
plt.scatter(x2, y2, c=scores2, cmap='Reds')
plt.colorbar(label='Score')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Outliers')
plt.show()

# Only show the inliers scores < 0.9
x3 = [row['x'] for row in data if row['score'] < sensitivity]
y3 = [row['y'] for row in data if row['score'] < sensitivity]
scores3 = [row['score'] for row in data if row['score'] < sensitivity]

# Create scatter plot
plt.scatter(x3, y3, c=scores3, cmap='Reds')
plt.colorbar(label='Score')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inliers')
plt.show()
