# SLAB QICK Calibration

A comprehensive package for calibrating and characterizing superconducting qubits using the QICK (Quantum Instrumentation Control Kit) framework.

## Overview

This package provides tools and utilities for calibrating and characterizing superconducting qubits using the QICK framework. It includes a wide range of experiments for measuring qubit parameters, optimizing control pulses, and characterizing qubit coherence. The package has been converted to a namespace package to allow importing functions and modules directly.

Key features:
- Single qubit characterization experiments (resonator spectroscopy, qubit spectroscopy, Rabi, T1, T2)
- Two qubit experiments
- Automated calibration workflows
- Data management and analysis tools
- Configuration management

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
    - `resonator_spectroscopy.py`: Characterizes readout resonators
    - `pulse_probe_spectroscopy.py`: Measures qubit transition frequencies
    - `rabi.py`: Calibrates qubit control pulses
    - `t1.py`: Measures energy relaxation time
    - `t2.py`: Measures phase coherence time (Ramsey and Echo)
    - `single_shot.py`: Optimizes single-shot readout
    - `stark_spectroscopy.py`: Measures AC Stark shifts
    - `active_reset.py`: Implements active qubit reset
    - `tof_calibration.py`: Calibrates time of flight for readout
  - `two_qubit/`: Two qubit experiments
    - `rabi_2q.py`: Two-qubit Rabi oscillations
    - `t1_2q.py`: Two-qubit T1 measurements
- `gen/`: Generated code and base classes for QICK experiments
  - `qick_experiment.py`: Base classes for QICK experiments
  - `qick_program.py`: Base classes for QICK programs

For detailed documentation on the experiments, see [README_experiments.md](README_experiments.md).

## Testing

You can run the test script to verify that the package is installed correctly:

```bash
python test_import.py
