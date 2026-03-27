#!/bin/bash
# EyeCursor Launcher
# This script activates the virtual environment and runs the app

cd "$(dirname "$0")"
source venv/bin/activate
python3 main.py
