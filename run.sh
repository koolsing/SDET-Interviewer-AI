#!/bin/bash

# Simple runner script for SDET Interview Coach

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment (venv) not found."
    echo "Please run ./setup.sh first to install dependencies."
    exit 1
fi

# Activate venv and run main.py
source venv/bin/activate
python3 main.py "$@"
