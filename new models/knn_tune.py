import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "./DATA/knn_tune_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
    target = mnist.target.astype(np.uint8)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Define hyperparameters to tune
    param_grid = {'n_neighbors': [3, 5, 7],
                  'weights': ['uniform', 'distance'],
                  'p': [1, 2]}  # p=1 for Manhattan distance, p=2 for Euclidean distance

    # Create GridSearchCV object
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=3, n_jobs=-1, verbose=2)

    # Perform hyperparameter tuning and training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Make predictions on the test set
    # y_pred = grid_search.predict(X_test)

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")
    print(f"Best Hyperparameters: {best_params}")

except Exception as e:
    print(f"An error occurred: {e}")
