import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Suppress Python warnings
warnings.filterwarnings('ignore')

try:
    # Generate a filename for the output file
    filename = "./DATA/dt_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

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

    # Define the Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree model
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Make predictions on the test set
    # y_pred = dt_model.predict(X_test)

    # Evaluate the accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy on the test set: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")
