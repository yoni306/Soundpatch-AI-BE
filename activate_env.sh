#!/bin/bash
# Activation script for the virtual environment

echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "You can now run your Python scripts."
echo ""
echo "To deactivate, run: deactivate"
echo ""
echo "Available packages:"
pip list | head -10
