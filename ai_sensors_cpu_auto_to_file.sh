#!/bin/bash

# Check if an output file name is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 output_file_name"
    exit 1
fi

# Use the argument as the output file name
output_file="$1"

# Start the monitoring loop
while true; do
    # Get the current date and time
    date_info=$(date)

    # Get CPU Utilization
    cpu_utilization=$(mpstat 1 1 | awk '/Average:/ {print $3 "%"}')

    # Get CPU Temperature
    # This requires lm-sensors to be installed and configured *sudo apt install lm-sensors*
    cpu_temp=$(sensors | grep 'Tctl:' | awk '{print $2}')

    # Write the information to the file
    echo "----------------------------------------" >> $output_file
    echo "Date and Time: $date_info" >> $output_file
    echo "CPU Utilization: $cpu_utilization" >> $output_file
    echo "CPU Temperature: $cpu_temp" >> $output_file

    # Wait for 1 seconds
    sleep 1
done
