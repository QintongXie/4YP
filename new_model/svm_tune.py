import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "./DATA/svm_tune_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset with parser='liac-arff'
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='liac-arff')
    data = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
    target = mnist.target.astype(np.uint8)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Define a few sets of hyperparameters to try
    hyperparams_to_try = [
        {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 1, 'kernel': 'linear'}
    ]

    best_accuracy = 0
    best_params = None

    # Train SVM models with different hyperparameter combinations
    for params in hyperparams_to_try:
        clf = SVC(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Hyperparameters: {params}, Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Print the best hyperparameters found
    print("Best hyperparameters found:")
    print(best_params)
    print("Best accuracy on test set:")
    print(best_accuracy)

except Exception as e:
    print(f"An error occurred: {e}")

