import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Additional TensorFlow logging suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    # Generate a filename for the output file
    filename = "./DATA/cnn_tune_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "./ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Add a channel dimension to the data (for Conv2D layer)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Define a function to build the model
    def build_model(hp):
        model = Sequential()
        model.add(Conv2D(hp.Int('conv1_units', min_value=32, max_value=256, step=32),
                         (3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(hp.Int('conv2_units', min_value=32, max_value=256, step=32),
                         (3, 3),
                         activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(hp.Int('dense_units', min_value=64, max_value=512, step=64), activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # Instantiate the tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,  # Number of hyperparameter combinations to try
        directory='my_tuning_dir',  # Directory to save the tuning results
        project_name='mnist_tuning')  # Name of the tuning project

    # Search for the best hyperparameter configuration
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Train the model
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")

    # Evaluate the model on the test set
    # y_pred = np.argmax(model.predict(X_test), axis=1)

except Exception as e:
    print(f"An error occurred: {e}")
