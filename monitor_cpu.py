import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow import keras
import psutil
import time
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1, parser='auto')
data = mnist.data.astype(np.uint8)
target = mnist.target.astype(np.uint8)

# Normalize pixel values to be between 0 and 1
data /= 255.0

'''
# SVM
# Use a subset (e.g., 20% of the data)
data_subset, _, target_subset, _ = train_test_split(data, target, train_size=0.2, random_state=42)

# Split the dataset into training and testing sets
X_train, _, y_train, _ = train_test_split(data_subset, target_subset, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = svm.SVC()

# Start monitoring CPU utilization
process = psutil.Process()

# Store data for plotting
cpu_utilization_data = []
cpu_frequency_data = []
time_data = []

# Train the SVM model and monitor CPU utilization and frequency during training
start_time = time.time()

# Number of training epochs
num_epochs = 20

for epoch in range(num_epochs):
    # Train the model for one epoch
    clf.fit(X_train, y_train)

    # Monitor CPU utilization and frequency during training
    cpu_percent = process.cpu_percent()
    cpu_freq = psutil.cpu_freq().current

    print(f"Epoch {epoch + 1} | Time: {time.time() - start_time:.2f} s | "
          f"CPU Utilization: {cpu_percent:.2f}% | CPU Frequency: {cpu_freq:.2f} MHz")

    # Store data for plotting
    cpu_utilization_data.append(cpu_percent)
    cpu_frequency_data.append(cpu_freq)
    time_data.append(time.time() - start_time)

    time.sleep(1)


# Random Forest
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Start monitoring CPU utilization
process = psutil.Process()

# Store data for plotting
cpu_utilization_data = []
cpu_frequency_data = []
time_data = []

# Train the Random Forest model and monitor CPU utilization and frequency during training
start_time = time.time()

# Number of training iterations
num_iterations = 20

for iteration in range(num_iterations):
    # Train the model for one iteration (you can adjust this based on your training procedure)
    clf.fit(X_train, y_train)

    # Monitor CPU utilization and frequency during training
    cpu_percent = process.cpu_percent()
    cpu_freq = psutil.cpu_freq().current

    print(f"Iteration {iteration + 1} | Time: {time.time() - start_time:.2f} s | "
          f"CPU Utilization: {cpu_percent:.2f}% | CPU Frequency: {cpu_freq:.2f} MHz")

    # Store data for plotting
    cpu_utilization_data.append(cpu_percent)
    cpu_frequency_data.append(cpu_freq)
    time_data.append(time.time() - start_time)

    time.sleep(1)
'''

# Neural Network
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Start monitoring CPU utilization
process = psutil.Process()

# Store data for plotting
cpu_utilization_data = []
cpu_frequency_data = []
time_data = []

# Train the neural network and monitor CPU utilization and frequency during training
start_time = time.time()

# Number of training epochs
num_epochs = 20

for epoch in range(num_epochs):
    # Train the model for one epoch
    history = model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=0)

    # Monitor CPU utilization and frequency during training
    cpu_percent = process.cpu_percent()
    cpu_freq = psutil.cpu_freq().current

    print(f"Epoch {epoch + 1} | Time: {time.time() - start_time:.2f} s | "
          f"CPU Utilization: {cpu_percent:.2f}% | CPU Frequency: {cpu_freq:.2f} MHz | "
          f"Validation Accuracy: {history.history['val_accuracy'][0]:.4f}")

    # Store data for plotting
    cpu_utilization_data.append(cpu_percent)
    cpu_frequency_data.append(cpu_freq)
    time_data.append(time.time() - start_time)

    time.sleep(1)


# Plot CPU utilization and frequency separately after training
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_data, cpu_utilization_data, label="CPU Utilization")
plt.scatter(time_data, cpu_utilization_data, color='red', marker='o', label='Marked Point (Utilization)')
plt.xlabel("Time (seconds)")
plt.ylabel("CPU Utilization (%)")
plt.title("CPU Utilization vs Time (During Training)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_data, cpu_frequency_data, label="CPU Frequency", linestyle='dashed')
plt.scatter(time_data, cpu_frequency_data, color='blue', marker='x', label='Marked Point (Frequency)')
plt.xlabel("Time (seconds)")
plt.ylabel("CPU Frequency (MHz)")
plt.title("CPU Frequency vs Time (During Training)")
plt.legend()

plt.tight_layout()
plt.show()
