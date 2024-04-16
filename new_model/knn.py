import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "./DATA/knn_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset with parser='liac-arff'
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='liac-arff')
    data = mnist.data.astype(np.uint8)
    target = mnist.target.astype(np.uint8)

    # Normalize pixel values to be between 0 and 1
    data = data.astype(np.float64) / 255.0

    # Flatten the images
    data_flatten = data.reshape((data.shape[0], -1))

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data_flatten, target, test_size=0.2, random_state=42)

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the KNN model
    start_time = time.time()
    knn_classifier.fit(X_train, y_train)
    end_time = time.time()

    # Stop monitoring CPU statistics
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Evaluate the model on the test set
    # y_pred = knn_classifier.predict(X_test)

except Exception as e:
    print(f"An error occurred: {e}")
