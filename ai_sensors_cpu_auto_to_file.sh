#!/bin/bash

# Specify the output file
output_file="cpu_monitor_output.txt"

# Start the monitoring loop
while true; do
    # Get the current date and time
    date_info=$(date)

    # Get CPU Utilization
    cpu_utilization=$(mpstat 1 1 | awk '/Average:/ {print $3 "%"}')

    # Get CPU Frequency
    cpu_max_freq=$(lscpu | grep "CPU max MHz" | awk '{print $4}')
    cpu_min_freq=$(lscpu | grep "CPU min MHz" | awk '{print $4}')

    # Get CPU Temperature
    # This requires lm-sensors to be installed and configured *sudo apt install lm-sensors*
    cpu_temp=$(sensors | grep 'Tctl:' | awk '{print $2}')

    # Write the information to the file
    echo "----------------------------------------" >> $output_file
    echo "Date and Time: $date_info" >> $output_file
    echo "CPU Utilization: $cpu_utilization" >> $output_file
    echo "CPU Max Frequency: ${cpu_max_freq} MHz" >> $output_file
    echo "CPU Min Frequency: ${cpu_min_freq} MHz" >> $output_file
    echo "CPU Temperature: $cpu_temp" >> $output_file

    # Wait for 5 seconds
    sleep 5
done

