import numpy as np

x = np.array([0.43, 2.83, 1.57, 1.24, 0.79, 2.32, 1.37, 1.71, 0.006, 1.85])
# Assuming y_true is the array of true values and y_pred is the array of predicted values
y_true = np.array([0.56, 11.13, 3.39, 2.2, 1.05, 7.43, 2.62, 4.01, 0.48, 4.72])  # replace with your true values

theta_1a = np.array([-4.9817, 5.5399])
theta_1b = np.array([-0.1750, 0.3396, 1.29])
theta_1c = np.array([-6.3545, 12.9432, -7.8985, 2.8502, -0.3193])

y_pred_1_a = sum(th * x**i for i, th in enumerate(theta_1a))
y_pred_1_b = sum(th * x**i for i, th in enumerate(theta_1b))
y_pred_1_c = sum(th * x**i for i, th in enumerate(theta_1c))


# Calculate MAE
mae_1_a = np.mean(np.abs(y_true - y_pred_1_a))
mae_1_b = np.mean(np.abs(y_true - y_pred_1_b))
mae_1_c = np.mean(np.abs(y_true - y_pred_1_c))

# Calculate MSE
mse_1_a = np.mean((y_true - y_pred_1_a)**2)
mse_1_b = np.mean((y_true - y_pred_1_b)**2)
mse_1_c = np.mean((y_true - y_pred_1_c)**2)

# Calculate RMSE
rmse_1_a = np.sqrt(mse_1_a)
rmse_1_b = np.sqrt(mse_1_b)
rmse_1_c = np.sqrt(mse_1_c)

print(f"MAE 1a: {mae_1_a}")
print(f"MSE 1a: {mse_1_a}")
print(f"RMSE 1a: {rmse_1_a}")
print(f"MAE 1b: {mae_1_b}")
print(f"MSE 1b: {mse_1_b}")
print(f"RMSE 1b: {rmse_1_b}")
print(f"MAE 1c: {mae_1_c}")
print(f"MSE 1c: {mse_1_c}")
print(f"RMSE 1c: {rmse_1_c}")