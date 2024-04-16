import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    filename = "./DATA/knn_cifar_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the CIFAR-10 dataset using tf.keras.datasets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Flatten the images
    X_train_flatten = X_train.reshape((X_train.shape[0], -1))
    X_test_flatten = X_test.reshape((X_test.shape[0], -1))

    # Convert data types to uint8
    X_train_flatten = X_train_flatten.astype(np.uint8)
    X_test_flatten = X_test_flatten.astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    # Create a KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the KNN model
    start_time = time.time()
    knn_classifier.fit(X_train_flatten, y_train)
    end_time = time.time()

    # Stop monitoring CPU statistics
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Evaluate the model on the test set
    # y_pred = knn_classifier.predict(X_test_flatten)

except Exception as e:
    print(f"An error occurred: {e}")
