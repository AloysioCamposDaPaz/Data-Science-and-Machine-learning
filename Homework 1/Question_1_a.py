import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1.65, 2.15, 1.81, 1.63, 1.27, 1.94, 1.31, 2.68, 2.89, 1.15]) #1.15
y = np.array([3.78, 6.70, 4.69, 3.72, 2.40, 5.25, 2.80, 9.85, 11.67, 1.70])

# Means of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculating theta_1 and theta_0
theta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
theta_0 = y_mean - theta_1 * x_mean

print(f"theta_0: {theta_0}")
print(f"theta_1: {theta_1}")

# Predicted y values
y_pred = theta_0 + theta_1 * x

# Plotting the data points
plt.scatter(x, y, color='red', label='Data points')

# Plotting the regression line
plt.plot(x, y_pred, color='blue', label='Regression line')

# Labeling axes
plt.xlabel('x')
plt.ylabel('y')

# Adding a legend to explain the markers
plt.legend()

# Display the plot
plt.show()