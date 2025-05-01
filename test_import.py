"""
Test script to verify that the slab_qick_calib package can be imported correctly
"""

# Import the package
import slab_qick_calib

# Import specific modules
from slab_qick_calib import calib
from slab_qick_calib import exp_handling
from slab_qick_calib import experiments
from slab_qick_calib import gen

# Print the version
print(f"slab_qick_calib version: {slab_qick_calib.__version__}")

# List available modules
print("\nAvailable modules:")
print("- calib")
print("- exp_handling")
print("- experiments")
print("- gen")

# Example of importing specific modules
print("\nExample imports:")
print("from slab_qick_calib.calib import qubit_tuning")
print("from slab_qick_calib.experiments.single_qubit import resonator_spectroscopy")
print("from slab_qick_calib.gen import qick_experiment")
