# slab_qick_calib

Version control for slab experiments to be running on rfsoc.

## Overview

This package provides tools and utilities for calibrating qubits using the QICK (Quantum Instrumentation Control Kit) framework. It has been converted to a namespace package to allow importing functions and modules directly.

## Installation

### Development Installation

For development, you can install the package in development mode:

```bash
# Clone the repository
git clone <repository-url>
cd slab_qick_calib

# Install in development mode
pip install -e .
```

### Regular Installation

```bash
pip install .
```

## Usage

After installation, you can import the package and its modules:

```python
# Import the package
import slab_qick_calib

# Import specific modules
from slab_qick_calib import calib
from slab_qick_calib import exp_handling
from slab_qick_calib import experiments
from slab_qick_calib import gen

# Import specific functions or classes
from slab_qick_calib.calib import qubit_tuning
from slab_qick_calib.experiments.single_qubit import resonator_spectroscopy
from slab_qick_calib.gen import qick_experiment
```

## Package Structure

- `calib/`: Calibration modules for qubit tuning
- `exp_handling/`: Experiment handling modules for data management and analysis
- `experiments/`: Experiment implementations
  - `single_qubit/`: Single qubit experiments
  - `two_qubit/`: Two qubit experiments
- `gen/`: Generated code and base classes for QICK experiments

## Testing

You can run the test script to verify that the package is installed correctly:

```bash
python test_import.py
