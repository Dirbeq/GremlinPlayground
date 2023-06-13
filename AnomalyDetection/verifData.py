import json
import matplotlib.pyplot as plt

# Load data
with open('INNE.json') as f:
    data = json.load(f)

# Extract x, y, and score values
x = [row[0] for row in data]
y = [row[1] for row in data]
scores = [row[2] for row in data]

# Create scatter plot
plt.scatter(x, y, c=scores, cmap='viridis')
plt.colorbar(label='Score')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
