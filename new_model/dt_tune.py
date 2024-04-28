import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "../DATA/dt_tune_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "../monitor/ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset with parser='liac-arff'
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='liac-arff')
    data = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
    target = mnist.target.astype(np.uint8)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Flatten the labels (assuming y_train and y_test are 2D arrays)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Define the Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)

    # Define hyperparameters for tuning
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=3)

    # Train the Decision Tree model with hyperparameter tuning
    start_time = time.time()
    grid_search.fit(X_train, y_train_flat)
    end_time = time.time()

    print("Best parameters: ", grid_search.best_params_)

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")
    # print("Accuracy on the test set: {:.4f}".format(grid_search.best_estimator_.score(X_test, y_test_flat)))

except Exception as e:
    print(f"An error occurred: {e}")
