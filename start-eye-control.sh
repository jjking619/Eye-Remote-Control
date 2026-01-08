#!/bin/bash

# Eye Remote Control Start Script
# This script activates the virtual environment and starts the application

cd /home/pi/eye-remote-control || exit 1

# Activate virtual environment
source ../mediapipe_env/bin/activate

# Start the application
python3 main.py