#!/usr/bin/env python
"""
Simplified test to verify the package structure.
This test doesn't require Detectron2 to be properly installed.
"""
import os
import sys

def main():
    # Check that all expected files exist
    files = [
        "README.md",
        "__init__.py",
        "demo.py",
        "example.py",
        "faster_rcnn_bg.py",
        "inference.py",
        "requirements.txt",
        "setup.py",
        "test.png"
    ]
    
    missing_files = []
    for file in files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: The following files are missing: {', '.join(missing_files)}")
        return False
    else:
        print("All expected files are present.")
    
    # Check imports
    try:
        import torch
        print("PyTorch is available.")
    except ImportError:
        print("WARNING: PyTorch is not installed.")
    
    try:
        import cv2
        print("OpenCV is available.")
    except ImportError:
        print("WARNING: OpenCV is not installed.")
    
    try:
        import numpy
        print("NumPy is available.")
    except ImportError:
        print("WARNING: NumPy is not installed.")
    
    print("\nPackage structure verification complete.")
    print("To test the full functionality, make sure Detectron2 is properly installed.")
    print("Then run the example.py or demo.py script with appropriate arguments.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 