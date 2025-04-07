#!/usr/bin/env python3
"""
Setup script for Marine Species Analysis
---------------------------------------
This script installs the required packages for running the marine species analyses.
"""

import sys
import subprocess
import os

def install_requirements():
    print("Installing required packages from requirements.txt...")
    try:
        # Use the same Python executable that's running this script
        python_executable = sys.executable
        subprocess.check_call([python_executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed all required packages!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print("Try running 'pip install -r requirements.txt' manually.")
        return False
    return True

def create_directories():
    print("Creating necessary directories...")
    os.makedirs('analysis_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    print("Directories created successfully!")

if __name__ == "__main__":
    print("Setting up environment for Marine Species Analysis...")
    if install_requirements():
        create_directories()
        print("\nSetup complete! You can now run the analyses with:")
        print("python run_all_analyses.py")
    else:
        print("\nSetup incomplete due to errors.")