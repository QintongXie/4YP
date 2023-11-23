# Project: Predicting AI Workloads
A repository for AI workloads prediction.

## Logs:
### 1. Monitor CPU information simply. (ai_sensors_cpu_auto_to_file.sh)
#### How It Works
The script captures the output of each command into variables.

It then writes these variables, along with some descriptive text, to a specified file (cpu_monitor_output.txt in this case).

The >> operator is used to append the data to the file.

The script will run indefinitely and write the information to the file every 5 seconds until you manually stop it with Ctrl+C.

#### Running the Script
Save this script into a file (e.g., ai_sensors_cpu_auto_to_file.sh).

Make it executable: *chmod +x ai_sensors_cpu_auto_to_file.sh*.

Run the script: *./ai_sensors_cpu_auto_to_file.sh*.

This script will continuously append the CPU data to cpu_monitor_output.txt. You can open this file in any text editor to view the logged information. If you want to log the data into a different file, simply change the output_file variable in the script.
