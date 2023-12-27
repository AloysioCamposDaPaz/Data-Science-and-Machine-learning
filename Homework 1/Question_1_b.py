import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1.65, 2.15, 1.81, 1.63, 1.27, 1.94, 1.31, 2.68, 2.89, 1.15])
y = np.array([3.78, 6.70, 4.69, 3.72, 2.40, 5.25, 2.80, 9.85, 11.67, 1.70])

# Construct the X matrix with [1, x, x^2] for each sample
X = np.vstack((np.ones_like(x), x, x**2)).T

# Solve for theta using the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"theta_0: {theta[0]}")
print(f"theta_1: {theta[1]}")
print(f"theta_2: {theta[2]}")

# Generate smooth x values for plotting the curve
x_smooth = np.linspace(x.min(), x.max(), 1000)

# Generate the predicted y values based on the smooth x values
y_smooth = theta[0] + theta[1]*x_smooth + theta[2]*x_smooth**2

# Plotting the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plotting the curve for the quadratic model
plt.plot(x_smooth, y_smooth, color='blue', label='Quadratic model')

# Labeling axes
plt.xlabel('x')
plt.ylabel('y')

# Adding a legend to identify the data points and curve
plt.legend()

# Displaying the plot
plt.show()
