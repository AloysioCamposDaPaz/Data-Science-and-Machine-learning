import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1.65, 2.15, 1.81, 1.63, 1.27, 1.94, 1.31, 2.68, 2.89, 1.15])
y = np.array([3.78, 6.70, 4.69, 3.72, 2.40, 5.25, 2.80, 9.85, 11.67, 1.70])

# Construct the X matrix with columns for each power of x from 0 to 4
X = np.column_stack([x**i for i in range(5)])

# Solve for theta using the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# Display the theta values
for i, th in enumerate(theta):
    print(f"theta_{i}: {th}")

# Generate smooth x values for plotting the curve
x_smooth = np.linspace(x.min(), x.max(), 1000)

# Generate the predicted y values based on the smooth x values
y_smooth = sum(th * x_smooth**i for i, th in enumerate(theta))

# Plotting the original data points
plt.scatter(x, y, color='red', label='Data points')

# Plotting the curve for the model
plt.plot(x_smooth, y_smooth, color='blue', label='Model')

# Labeling axes
plt.xlabel('x')
plt.ylabel('y')

# Adding a legend to identify the data points and curve
plt.legend()

# Displaying the plot
plt.show()
