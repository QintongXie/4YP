import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
import tensorflow as tf

# Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Additional TensorFlow logging suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    # Generate a filename for the output file
    filename = "./DATA/svm_cifar_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten the images
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Convert labels to 1D array
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Split the data into training and testing sets
    # Example: Use only the first 1000 samples for training
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_flat[:1000], y_train_flat[:1000], test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm_model = SVC()

    # Train the SVM model
    start_time = time.time()
    svm_model.fit(x_train_split, y_train_split)
    end_time = time.time()

    # Make predictions on the validation set
    y_val_pred = svm_model.predict(x_val_split)

    # Calculate validation accuracy
    # accuracy = accuracy_score(y_val_split, y_val_pred)

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

except Exception as e:
    print(f"An error occurred: {e}")
