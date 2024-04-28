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
    # Eagle House----------------------------------------------
    # cpu_temp1=$(sensors | grep 'temp1:' | awk '{print $2}')
    # cpu_temp2=$(sensors | grep 'temp2:' | awk '{print $2}')
    # cpu_temp3=$(sensors | grep 'temp3:' | awk '{print $2}')
    # Workstation----------------------------------------------
    cpu_temp1=$(sensors | grep 'Tctl:' | awk '{print $2}')


    # Write the information to the file
    echo "----------------------------------------" >> $output_file
    echo "Date and Time: $date_info" >> $output_file
    echo "CPU Utilization: $cpu_utilization" >> $output_file

    # Eagle House and Workstation----------------------------------------------
    echo "CPU Temperature1: $cpu_temp1" >> $output_file

    # Eagle House----------------------------------------------
    # echo "CPU Temperature2: $cpu_temp2" >> $output_file
    # echo "CPU Temperature3: $cpu_temp3" >> $output_file

    # Wait for 1 seconds
    sleep 1
done
