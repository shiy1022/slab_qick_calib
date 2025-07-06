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
from slab_qick_calib import analysis
from slab_qick_calib import helpers

# Import experiments as a group (common pattern)
import slab_qick_calib.experiments as meas

# Import specific functions or classes
from slab_qick_calib.calib import qubit_tuning, measure_func
from slab_qick_calib.experiments.single_qubit import resonator_spectroscopy
from slab_qick_calib.experiments.general import qick_experiment
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
```

## Package Structure

- `analysis/`: Data analysis and fitting tools
  - `fitting.py`: Curve fitting functions for experiment data
  - `qubit_params.py`: Qubit parameter extraction and analysis
  - `time_series.py`: Time series data analysis tools
- `calib/`: Calibration modules for qubit tuning
  - `measure_func.py`: Measurement functions for calibration, currently chi and temperature
  - `qubit_tuning.py`: Automated qubit tuning workflows
  - `readout_helpers.py`: Readout utilities
  - `time_tracking.py`: Time tracking for experiments
- `configs/`: Configuration files for different systems
  - Various `.yml` and `.cfg` files for instrument configurations
- `exp_handling/`: Experiment handling modules for data management and analysis (slab files, so that you don't need to install slab)
  - `dataanalysis.py`: Data analysis utilities
  - `datamanagement.py`: Data storage and retrieval
  - `experiment.py`: Base experiment classes
  - `instrumentmanager.py`: Instrument management and control
- `experiments/`: Experiment implementations
  - `general/`: Base classes for QICK experiments
    - `qick_experiment.py`: Base classes for single qubit QICK experiments
    - `qick_experiment_2q.py`: Base classes for two qubit QICK experiments
    - `qick_program.py`: Base classes for QICK programs
  - `single_qubit/`: Single qubit experiments
    - `active_reset.py`: Checks of active qubit reset parameters, mostly not used
    - `pulse_probe_spectroscopy.py`: Measures qubit transition frequencies
    - `rabi.py`: Calibrates qubit control pulses
    - `resonator_spectroscopy.py`: Characterizes readout resonators
    - `single_shot.py`: Single-shot readout fidelity
    - `single_shot_opt.py`: Optimizes single-shot readout
    - `stark_spectroscopy.py`: Measures AC Stark shifts, still writing up.
    - `t1.py`: Measures energy relaxation time
    - `t1_cont.py`: Fast continuous T1 measurements 
    - `t1_stark.py`: T1 measurements with Stark shifts of qubit
    - `t2.py`: Measures phase coherence time (Ramsey and Echo)
    - `t2_ramsey_stark.py`: Ramsey T2 with Stark shifts
    - `tof_calibration.py`: Calibrates time of flight for readout
  - `two_qubit/`: Two qubit experiments
    - `rabi_2q.py`: Two-qubit Rabi oscillations
    - `t1_2q.py`: Two-qubit T1 measurements
- `helpers/`: Utility functions and configuration helpers
  - `config.py`: Configuration file handling
  - `handy.py`: General utility functions
  - `qick_check.py`: QICK system verification tools
- `notebooks/`: Example notebooks and tutorials
  - Various Jupyter notebooks demonstrating package usage

For detailed documentation on the experiments, see [README_experiments.md](README_experiments.md).
For detailed documentation on the config file, see [config_manual.md](config_manual.md).
For detailed documentation on the base classes, see [README_classes.md](README_classes.md).

## Testing

You can verify that the package is installed correctly by importing it in Python:

```python
import slab_qick_calib
print("Package imported successfully!")

# Test importing key modules
from slab_qick_calib import experiments, calib, exp_handling
print("Core modules imported successfully!")
```
