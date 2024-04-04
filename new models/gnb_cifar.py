import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "./DATA/gnb_cifar_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Reshape and normalize pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten the images for Gaussian Naive Bayes
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Apply Principal Component Analysis (PCA) for dimensionality reduction
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    # Flatten the labels (assuming y_train and y_test are 2D arrays)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Define the Gaussian Naive Bayes model
    model = GaussianNB()

    # Train the Gaussian Naive Bayes model on the PCA-transformed data
    start_time = time.time()
    model.fit(X_train_pca, y_train_flat)
    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Make predictions on the test set
    # y_pred = model.predict(X_test_pca)

    # Evaluate the accuracy
    # accuracy = accuracy_score(y_test_flat, y_pred)
    # print(f"Accuracy on the test set: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")
