#!/bin/bash

# Directory containing the Python model files
model_dir="../new_model"

# Iterate through Python files in the model directory
for model_file in "${model_dir}"/*.py; do
    # Extract the model name from the file path (removing path and extension)
    model_name=$(basename "${model_file}" .py)
    
    # Run the command 10 times
    for ((n=1; n<=10; n++)); do
        # Define the output file name based on the model name and iteration number
        output_file="../data0427_log2_all1_workstation/${model_name}_cpu_log2_${n}.txt"
        echo "special python3 ${model_file}, times: ${n}"
        
        # Run the performance monitoring command and save the output for log2
        # python3 "${model_file}"
        perf stat -I 1000 -e cycles,instructions,cache-references,cache-misses,branches,branch-misses /usr/bin/python3 "${model_file}" > "${output_file}" 2>&1
    done
done

