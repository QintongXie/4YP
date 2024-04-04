import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

# Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Additional TensorFlow logging suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    # Generate a filename for the output file
    filename = "./DATA/dt_cifar_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Reshape and normalize pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten the images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Flatten the labels (assuming y_train and y_test are 2D arrays)
    y_train_flat = y_train.flatten()
    y_test_flat = y_test.flatten()

    # Define the Decision Tree model
    dt_model = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree model
    start_time = time.time()
    dt_model.fit(X_train_flat, y_train_flat)
    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Make predictions on the test set
    # y_pred = dt_model.predict(X_test_flat)

    # Evaluate the Decision Tree model
    # accuracy = accuracy_score(y_test_flat, y_pred)
    # print(f"Accuracy on the test set: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")
