#!/bin/bash

# Directory containing the Python model files
model_dir="./model"

# Iterate through Python files in the model directory
for model_file in "${model_dir}"/rf_cifar.py; do
    # Extract the model name from the file path (removing path and extension)
    model_name=$(basename "${model_file}" .py)
    
    # Run the command 10 times
    for ((n=1; n<=100; n++)); do
        # Define the output file name based on the model name and iteration number
        output_file="./4YP/data/${model_name}_cpu_log2_${n}.txt"
        
        # Run the performance monitoring command and save the output for log2
        sudo perf stat -I 1000 -e cycles,instructions,cache-references,cache-misses,branches,branch-misses python3 "${model_file}" > "${output_file}" 2>&1
    done
done

