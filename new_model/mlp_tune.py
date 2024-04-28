import os
import time
from datetime import datetime
import warnings
import subprocess
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
import keras_tuner
from kerastuner.tuners import RandomSearch

# Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Additional TensorFlow logging suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    # Generate a filename for the output file
    filename = "../DATA/mlp_tune_cpu_log1_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

    # Path to your Bash script
    bash_script_path = "../monitor/ai_sensors_cpu_auto_to_file.sh"

    # Start the Bash script with subprocess.Popen, passing the filename as an argument
    monitor_process = subprocess.Popen(['/bin/bash', bash_script_path, filename])

    # Load the MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
    target = mnist.target.astype(np.uint8)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(layers.Flatten(input_shape=(28, 28)))

        # Tune the number of units in the first Dense layer
        model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))

        # Tune the number of Dense layers and dropout rate
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.Dense(units=hp.Int(f'dense_{i}_units', min_value=32, max_value=512, step=32),
                                   activation='relu'))
            model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}_rate', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Define the tuner
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,  # You can increase this for more exhaustive search
        executions_per_trial=1,
        directory='my_dir',
        project_name='mnist_hyperparameter_tuning'
    )

    # Perform hyperparameter tuning
    tuner.search(X_train.reshape(-1, 28, 28), y_train, epochs=5, validation_split=0.2)

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Print the best hyperparameters
    print(f"Best Hyperparameters: {best_hps}")

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Train the best model
    start_time = time.time()
    best_model.fit(X_train.reshape(-1, 28, 28), y_train, epochs=10, validation_split=0.2, verbose=0)
    end_time = time.time()

    # Stop monitoring CPU utilization
    monitor_process.kill()

    # Calculate training time
    training_time = end_time - start_time
    print(f"Training completed. Training time: {training_time:.2f} seconds")
except Exception as e:
    print(f"An error occurred: {e}")
