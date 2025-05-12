# SLAB QICK Calibration Experiments

This document provides a guide to configuring and running various quantum experiments using the SLAB QICK calibration framework. The framework is designed for calibrating and characterizing superconducting qubits using the QICK (Quantum Instrumentation Control Kit) hardware.

## Table of Contents
- [Introduction](#introduction)
- [Experiment Configuration](#experiment-configuration)
- [Available Experiments](#available-experiments)
  - [Single Qubit Experiments](#single-qubit-experiments)
  - [Two Qubit Experiments](#two-qubit-experiments)
- [Common Experiment Workflows](#common-experiment-workflows)
- [Parameter Reference](#parameter-reference)

## Introduction

The SLAB QICK calibration framework provides a comprehensive set of experiments for characterizing and calibrating superconducting qubits. The framework is built on top of the QICK hardware platform and provides a high-level interface for running quantum experiments.

Before running experiments, make sure a nameserver is running on the network, the QICK board is connected to it, and the IP address in your configuration matches that of the nameserver.

## Experiment Configuration

Experiments are configured using parameter dictionaries that are passed to the experiment constructor. Each experiment has a set of default parameters that can be overridden by the user.

### Basic Configuration Setup

To set up a new configuration or data folder:

```python
# Set up paths
expt_path = 'C:\_Data\YourExperiment\2025-05-12\'
cfg_file = 'YourConfig.yml'

# Create new config and folders if needed
import os
import slab_qick_calib.config as config

new_config = False  # Set to True to create a new config
new_folder = False  # Set to True to create a new data folder

configs_dir = os.path.join(os.getcwd(), 'configs')
cfg_file_path = os.path.join(configs_dir, cfg_file)
images_dir = os.path.join(expt_path, 'images')
summary_dir = os.path.join(images_dir, 'summary')

if new_config or new_folder:
    if new_config:
        # Initialize config with 4 qubits, using 'full' type and 'bf1_soc' aliases
        config.init_config(cfg_file_path, 4, type='full', aliases='bf1_soc', t1=30)

    if not os.path.exists(expt_path):
        os.makedirs(expt_path)
        os.mkdir(images_dir)
        os.mkdir(summary_dir)
```

### Running Experiments

To run an experiment, you need to:
1. Load the configuration
2. Connect to instruments
3. Create a configuration dictionary
4. Initialize and run the experiment

```python
# Load configuration
cfg_path = os.path.join(os.getcwd(), 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

# Connect to instruments
im = InstrumentManager(ns_address=ip)
soc = QickConfig(im[auto_cfg['aliases']['soc']].get_cfg())

# Create configuration dictionary
cfg_dict = {'soc': soc, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

# Run an experiment (example: Resonator Spectroscopy)
qi = 0  # Qubit index
rspec = meas.ResSpec(cfg_dict, qi=qi, style='coarse', params={'start': 5000, 'span': 1700})
```

### Experiment Parameters

Each experiment accepts a `params` dictionary that allows you to customize the experiment. Common parameters include:

- `reps`: Number of repetitions for each experiment point
- `soft_avgs`: Number of software averages
- `expts`: Number of experiment points
- `start`: Start value for the swept parameter
- `span`: Span of the swept parameter
- `gain`: Pulse amplitude (DAC units)
- `length`: Pulse length (μs)
- `frequency`: Pulse frequency (MHz)
- `active_reset`: Whether to use active reset

Experiment-specific parameters are documented in the respective experiment classes.

## Available Experiments

### Single Qubit Experiments

#### Time of Flight (ToF) Calibration

**File**: `experiments/single_qubit/tof_calibration.py`

**Description**: Measures the time it takes for the signal to run through the wires. It determines the time in clock ticks that we should wait to make measurements.

**Key Parameters**:
- `readout_length`: Length of the readout pulse in μs
- `trig_offset`: Trigger offset in μs
- `frequency`: Readout frequency in MHz
- `check_e`: Whether to check the excited state

**Example**:
```python
tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
```

#### Resonator Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Measures the resonant frequency of the readout resonator when the qubit is in its ground state. It sweeps the readout pulse frequency and looks for the frequency with the maximum measured amplitude.

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `gain`: Gain of the readout resonator
- `style`: 'coarse' for wide frequency scan, 'fine' for narrow scan

**Example**:
```python
# Coarse scan to find resonators
rspecc = meas.ResSpec(cfg_dict, qi=0, style='coarse', params={'start': 5000, 'span': 1700})

# Fine scan to precisely measure a resonator
rspec = meas.ResSpec(cfg_dict, qi=0, params={'span': 5})
```

#### Resonator Power Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Finds a good value for gain to park your readout at until you run readout optimization. It performs a 2D sweep of frequency and power to identify where the resonator 'breaks' (transitions from e to f state).

**Key Parameters**:
- `rng`: Range for the gain sweep
- `span`: Frequency span (MHz)
- `f_off`: Frequency offset (MHz)
- `expts_gain`: Number of gain points

**Example**:
```python
rpowspec = meas.ResSpecPower(cfg_dict, qi=0, params={'rng': 300, 'span': 5, 'f_off': 1, 'expts_gain': 30})
```

#### Qubit Spectroscopy

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Identifies the qubit transition frequency by applying a probe pulse and measuring the qubit response. It can be used to find both the g-e and e-f transitions.

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `gain`: Probe pulse amplitude
- `length`: Probe pulse length (μs)
- `checkEF`: Whether to check the e-f transition
- `style`: 'coarse', 'medium', 'fine', or 'huge' for different scan ranges

**Example**:
```python
# Find g-e transition
qspec = meas.QubitSpec(cfg_dict, qi=0, style='medium')

# Find e-f transition
qspec_ef = meas.QubitSpec(cfg_dict, qi=0, style='medium', params={'checkEF': True})
```

#### Qubit Spectroscopy Power

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Performs a 2D sweep of frequency and power to map out the qubit response. Useful for identifying power-dependent frequency shifts and optimal drive powers.

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `max_gain`: Maximum gain value
- `expts_gain`: Number of gain points
- `rng`: Range for logarithmic gain sweep

**Example**:
```python
qspec_pow = meas.QubitSpecPower(cfg_dict, qi=0, params={'span': 40, 'expts': 100, 'max_gain': 0.4})
```

#### Rabi Experiment

**File**: `experiments/single_qubit/rabi.py`

**Description**: Measures Rabi oscillations by varying either the amplitude or length of a drive pulse. Used to calibrate π and π/2 pulses.

**Key Parameters**:
- `expts`: Number of amplitude/length points
- `gain`: Maximum pulse amplitude (for length sweep)
- `sigma`: Pulse width (for amplitude sweep)
- `sweep`: 'amp' or 'length' to specify what to sweep
- `checkEF`: Whether to check the e-f transition
- `pulse_type`: 'gauss' or 'const' for pulse shape

**Example**:
```python
# Amplitude Rabi
amp_rabi = meas.RabiExperiment(cfg_dict, qi=0)

# Length Rabi
len_rabi = meas.RabiExperiment(cfg_dict, qi=0, params={'sweep': 'length', 'type': 'const', 'sigma': 0.1, 'gain': 0.15})
```

#### Rabi Chevron

**File**: `experiments/single_qubit/rabi.py`

**Description**: Performs a 2D sweep of pulse amplitude/length and frequency to map out Rabi oscillations as a function of detuning.

**Key Parameters**:
- `span_f`: Frequency span (MHz)
- `expts_f`: Number of frequency points
- `sweep`: 'amp' or 'length' to specify what to sweep

**Example**:
```python
amp_rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=0, params={'span_f': 10})
```

#### T1 Experiment

**File**: `experiments/single_qubit/t1.py`

**Description**: Measures the energy relaxation time (T1) by exciting the qubit to the |1⟩ state and measuring its decay over time.

**Key Parameters**:
- `expts`: Number of wait time points
- `start`: Start wait time (μs)
- `span`: Total span of wait times (μs)
- `acStark`: Whether to apply AC Stark shift during wait time

**Example**:
```python
t1 = meas.T1Experiment(cfg_dict, qi=0)
```

#### T2 Ramsey Experiment

**File**: `experiments/single_qubit/t2.py`

**Description**: Measures the phase coherence time (T2*) using a Ramsey sequence (π/2 - wait - π/2).

**Key Parameters**:
- `expts`: Number of wait time points
- `start`: Start wait time (μs)
- `span`: Total span of wait times (μs)
- `ramsey_freq`: Frequency detuning for phase advancement (MHz)
- `experiment_type`: 'ramsey' for standard Ramsey experiment
- `checkEF`: Whether to check the e-f transition

**Example**:
```python
t2r = meas.T2Experiment(cfg_dict, qi=0)
```

#### T2 Echo Experiment

**File**: `experiments/single_qubit/t2.py`

**Description**: Measures the echo coherence time (T2E) using a Hahn echo sequence (π/2 - wait/2 - π - wait/2 - π/2).

**Key Parameters**:
- `expts`: Number of wait time points
- `start`: Start wait time (μs)
- `span`: Total span of wait times (μs)
- `experiment_type`: 'echo' for echo experiment
- `num_pi`: Number of π pulses (1 for standard echo)

**Example**:
```python
t2e = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo'})
```

#### Single Shot Readout

**File**: `experiments/single_qubit/single_shot.py`

**Description**: Performs single-shot readout measurements to characterize readout fidelity and optimize readout parameters.

**Key Parameters**:
- `shots`: Number of single-shot measurements
- `check_e`: Whether to check the excited state
- `check_f`: Whether to check the second excited state
- `active_reset`: Whether to use active reset

**Example**:
```python
shot = meas.HistogramExperiment(cfg_dict, qi=0, params={'shots': 20000})
```

#### Single Shot Optimization

**File**: `experiments/single_qubit/single_shot.py`

**Description**: Optimizes readout parameters (frequency, gain, length) to maximize readout fidelity.

**Key Parameters**:
- `expts_f`: Number of frequency points
- `expts_gain`: Number of gain points
- `expts_len`: Number of readout length points
- `span_f`: Frequency span (MHz)
- `span_gain`: Gain span
- `span_len`: Readout length span (μs)

**Example**:
```python
shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=0, params={'expts_f': 5, 'expts_gain': 7, 'expts_len': 5})
```

#### Stark Spectroscopy

**File**: `experiments/single_qubit/stark_spectroscopy.py`

**Description**: Measures the AC Stark shift of the qubit frequency as a function of drive power.

**Key Parameters**:
- Similar to QubitSpec, with additional Stark-specific parameters

**Example**:
```python
stark_spec = meas.StarkSpec(cfg_dict, qi=0, style='medium')
```

### Two Qubit Experiments

#### Two-Qubit Rabi

**File**: `experiments/two_qubit/rabi_2q.py`

**Description**: Performs Rabi oscillations on two qubits simultaneously.

**Key Parameters**:
- Similar to single-qubit Rabi, but with a list of two qubits

**Example**:
```python
rabi_2q = meas.Rabi_2Q(cfg_dict, qi=[0, 1])
```

#### Two-Qubit T1

**File**: `experiments/two_qubit/t1_2q.py`

**Description**: Measures T1 relaxation times for two qubits simultaneously.

**Key Parameters**:
- Similar to single-qubit T1, but with a list of two qubits

**Example**:
```python
t1_2q = meas.T1_2Q(cfg_dict, qi=[0, 1])
```

## Common Experiment Workflows

### Resonator and Qubit Characterization

1. **Time of Flight Calibration**:
   ```python
   tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
   ```

2. **Resonator Spectroscopy**:
   ```python
   # Coarse scan to find resonators
   rspecc = meas.ResSpec(cfg_dict, qi=0, style='coarse', params={'start': 5000, 'span': 1700})
   
   # Fine scan for precise measurement
   rspec = meas.ResSpec(cfg_dict, qi=0, params={'span': 5})
   
   # Update configuration with resonator frequency
   rspec.update(cfg_dict['cfg_file'])
   ```

3. **Resonator Power Spectroscopy**:
   ```python
   rpowspec = meas.ResSpecPower(cfg_dict, qi=0, params={'rng': 300, 'span': 5})
   ```

4. **Qubit Spectroscopy**:
   ```python
   # Find qubit frequency
   qspec = meas.QubitSpec(cfg_dict, qi=0, style='medium')
   
   # Update configuration with qubit frequency
   if qspec.status:
       auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data["best_fit"][2], 0)
   ```

### Qubit Coherence Measurements

1. **Rabi Oscillations**:
   ```python
   # Amplitude Rabi to calibrate pi pulse
   amp_rabi = meas.RabiExperiment(cfg_dict, qi=0)
   
   # Update pi pulse amplitude
   config.update_qubit(cfg_path, ('pulses', 'pi_ge', 'gain'), amp_rabi.data['pi_length'], 0)
   ```

2. **T1 Measurement**:
   ```python
   t1 = meas.T1Experiment(cfg_dict, qi=0)
   
   # Update T1 value
   t1.update(cfg_path)
   ```

3. **T2 Ramsey Measurement**:
   ```python
   t2r = meas.T2Experiment(cfg_dict, qi=0)
   
   # Update qubit frequency and T2 value
   if t2r.status:
       config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], 0)
       config.update_qubit(cfg_path, 'T2r', t2r.data['best_fit'][3], 0)
   ```

4. **T2 Echo Measurement**:
   ```python
   t2e = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo'})
   
   # Update T2E value
   if t2e.status:
       config.update_qubit(cfg_path, 'T2e', t2e.data['best_fit'][3], 0)
   ```

### Readout Optimization

1. **Single Shot Measurement**:
   ```python
   shot = meas.HistogramExperiment(cfg_dict, qi=0, params={'shots': 20000})
   
   # Update readout parameters
   shot.update(cfg_path)
   ```

2. **Single Shot Optimization**:
   ```python
   shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=0, params={'expts_f': 5, 'expts_gain': 7, 'expts_len': 5})
   
   # Update optimized readout parameters
   shotopt.update(cfg_dict['cfg_file'])
   ```

## Parameter Reference

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reps` | Number of repetitions for each experiment point | Varies by experiment |
| `soft_avgs` | Number of software averages | Varies by experiment |
| `expts` | Number of experiment points | Varies by experiment |
| `start` | Start value for the swept parameter | Varies by experiment |
| `span` | Span of the swept parameter | Varies by experiment |
| `gain` | Pulse amplitude (DAC units) | Varies by experiment |
| `length` | Pulse length (μs) | Varies by experiment |
| `frequency` | Pulse frequency (MHz) | Varies by experiment |
| `active_reset` | Whether to use active reset | False |
| `qubit` | List of qubit indices | [qi] |
| `qubit_chan` | Qubit channel for readout | From config |

### Experiment-Specific Parameters

Each experiment has additional parameters specific to its functionality. Refer to the experiment class documentation for details.
