import numpy as np
import GPy
import matplotlib.pyplot as plt

# Define the function to parse CPU log files
def parse_cpu_log(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()

    parsed_data = []

    for line in data:
        if not line.startswith('#') and not line.startswith('Training completed'):
            parts = line.split()
            time = float(parts[0])
            counts = int(parts[1].replace(',', ''))
            unit = parts[2]
            events = ' '.join(parts[3:])
            parsed_data.append({'time': time, 'counts': counts, 'unit': unit, 'events': events})

    return parsed_data

# Load and preprocess data from CPU log files
X_list = []
Y_list = []
empty_files = []

for i in range(1, 61):
    file_path = f"/Users/angela/Downloads/data0311/rf_cifar_cpu_log2_{i}.txt"
    events_list = parse_cpu_log(file_path)

    if events_list:
        event_counts = []
        for event in events_list:
            count = float(event['counts'])
            event_counts.append(count)
        X_list.append(np.arange(len(event_counts)).reshape(-1, 1))
        Y_list.append(np.array(event_counts).reshape(-1, 1))
    else:
        empty_files.append(file_path)

if empty_files:
    print("The following files have empty or all-zero data:")
    for file_path in empty_files:
        print(file_path)

if X_list and Y_list:
    X_data = np.concatenate(X_list, axis=0).astype(np.float64)
    Y_data = np.concatenate(Y_list, axis=0).astype(np.float64)

# Normalize input and output data
X_mean, X_std = np.mean(X_data, axis=0), np.std(X_data, axis=0)
Y_mean, Y_std = np.mean(Y_data, axis=0), np.std(Y_data, axis=0)

X_data_normalized = (X_data - X_mean) / X_std
Y_data_normalized = (Y_data - Y_mean) / Y_std

# Define the kernel for the Gaussian Process
kernel = GPy.kern.RBF(input_dim=1, ARD=True)

# Create the Gaussian Process Regression model with normalized data
model = GPy.models.GPRegression(X_data_normalized, Y_data_normalized, kernel)

# Optionally, optimize the hyperparameters of the model
model.optimize(messages=True)

# Predictions
# Assuming X_test contains the test data for time series
# Replace X_test with your test data
X_test = np.linspace(min(X_data_normalized), max(X_data_normalized), 100).reshape(-1, 1)
mean, var = model.predict(X_test)

# Denormalize predictions
mean_denormalized = mean * Y_std + Y_mean
var_denormalized = var * Y_std**2

# Optionally, plot the predictions
plt.plot(X_data, Y_data, 'kx', mew=2)
plt.plot(X_test, mean_denormalized)
plt.fill_between(X_test[:, 0], mean_denormalized[:, 0] - np.sqrt(var_denormalized[:, 0]),
                 mean_denormalized[:, 0] + np.sqrt(var_denormalized[:, 0]), alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Predicted Value')
plt.title('Predicted CPU Event Measurements')
plt.show()
