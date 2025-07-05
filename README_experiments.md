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

Before running experiments, make sure a nameserver is running on the network (on whatever computer you want to run it, open NameServer.ipynb and run with that computer's IP address). Then, on QICK board, go to pyro4 folder and open 01_server.ipynb, setting an alias for the board that you will use when connecting to the board on your client computer, and setting the ip address to that of the nameserver. This is also where you would specify the location of firmware.

## Experiment Configuration
## QICK Program vs. QICK Experiment

Experiments are configured using parameter dictionaries that are passed to the experiment constructor. Each experiment has a set of default parameters that can be overridden by the user.
In this framework, it's important to understand the distinction between a `QickProgram` and a `QickExperiment`:

### Basic Configuration Setup
- **`QickProgram`**: This is a lower-level class that directly interacts with the QICK's pulse processor. It is responsible for defining the pulse sequences, acquiring data, and managing the hardware registers. Each experiment has an associated `QickProgram` that handles the hardware communication.
- **`QickExperiment`**: This is a higher-level class that encapsulates the entire experimental procedure. It handles data processing, fitting, plotting, and saving. It uses a `QickProgram` to execute the pulse sequence on the hardware. When you run an experiment, you will typically interact with the `QickExperiment` class.

This separation of concerns allows for a modular and extensible framework where the high-level experiment logic is decoupled from the low-level hardware control.

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
- `rounds`: Number of software averages
- `expts`: Number of experiment points
- `start`: Start value for the swept parameter
- `span`: Span of the swept parameter
- `gain`: Pulse amplitude (DAC units)
- `length`: Pulse length (μs)
- `frequency`: Pulse frequency (MHz)
- `active_reset`: Whether to use active reset

Experiment-specific parameters are documented in the respective experiment classes.

All times are in us. All gains are out of 1. 

## Available Experiments

### Single Qubit Experiments

#### Time of Flight (ToF) Calibration

**File**: `experiments/single_qubit/tof_calibration.py`

**Description**: Measures the time it takes for the signal to run through the cables, plus latency. It determines the time that we should wait to make measurements. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
Not automated, need to figure out trig_offset by eye.

**Purpose**: Run this calibration when the wiring of the setup is changed. By calibrating this delay, we ensure that data acquisition starts at the optimal time when the signal actually arrives at the detector, avoiding dead time or missed signals.

**Parameters**:
- `rounds`: Number of software averages for the measurement
- `readout_length [us]`: Length of the readout pulse
- `trig_offset [us]`: Current trigger offset for the ADC
- `gain [DAC units]`: Amplitude of the readout pulse
- `frequency [MHz]`: Frequency of the readout pulse
- `reps`: Number of averages per point
- `qubit`: List containing the qubit index to calibrate
- `phase`: Phase of the readout pulse
- `final_delay [us]`: Final delay after the pulse sequence
- `check_e`: Whether to excite qubit to |1⟩ state before measurement
- `use_readout`: Whether to use current readout parameters (gain and phase)

**Example**:
```python
tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
```

**2D Time of Flight Calibration**:

The package also provides a 2D Time of Flight calibration experiment (`ToF2D`) that performs multiple measurements over time. This can be useful for performing many iterations of the readout trace.

```python
tof_2d = meas.ToF2D(cfg_dict=cfg_dict, qi=0, params={'expts_count': 500})
```

#### Resonator Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Measures the resonant frequency of the readout resonator by sweeping the readout pulse frequency and looking for the frequency with the maximum measured amplitude. The resonator frequency is stored in the parameter cfg.device.readout.frequency.

The module includes:
- `ResSpecProgram`: Defines the pulse sequence for the resonator spectroscopy experiment
- `ResSpec`: Main experiment class for resonator spectroscopy
- `ResSpecPower`: 2D version that sweeps both frequency and power
- `ResSpec2D`: 2D version for repeated measurements

Note that harmonics of the clock frequency will show up as "infinitely" narrow peaks!

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz) 
- `center`: Center frequency (MHz) - alternative to `start'
- `expts`: Number of frequency points
- `gain`: Gain of the readout resonator
- `length`: Length of the readout pulse
- `final_delay`: Delay time between repetitions in μs
- `pulse_e`: Boolean to add e pulse prior to measurement (excite qubit)
- `pulse_f`: Boolean to add f pulse prior to measurement (excite to 2nd level)
- `style`: 'coarse' for wide frequency scan, 'fine' for narrow scan
- `long_pulse`: Whether to use a long readout pulse, requiring 'periodic' mode where readout pulse occuring entire time, when pulse length exceeds hardware limit. 
- `kappa`: Resonator linewidth

**Example**:
```python
# Coarse scan to find resonators
rspecc = meas.ResSpec(cfg_dict, qi=0, style='coarse', params={'start': 5000, 'span': 1700})

# Fine scan to precisely measure a resonator
rspec = meas.ResSpec(cfg_dict, qi=0, params={'span': 5})

# Measure resonator with qubit in excited state (to measure dispersive shift)
rspec_e = meas.ResSpec(cfg_dict, qi=0, params={'pulse_e': True})
```

#### Resonator Power Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Performs a 2D sweep of both readout frequency and power to map out how the resonator response changes with power. This allows measurement of the Lamb shift and other power-dependent effects. It's useful for finding a good value for gain to park your readout at until you run readout optimization.

**Key Parameters**:
- `rng`: Range for the gain sweep, going from max_gain to max_gain/rng (default: 100)
- `max_gain`: Maximum gain value (default: qubit.max_gain)
- `expts_gain`: Number of gain points (default: 20)
- `span`: Frequency span (MHz)
- `f_off`: Frequency offset from resonant frequency in MHz (usually negative)
- `min_reps`: Minimum number of repetitions (default: 100)
- `log`: Whether to use logarithmic scaling for the gain sweep (default: True)
- `pulse_e`: Whether to apply a π pulse to excite the qubit before measurement

**Example**:
```python
# Standard power sweep
rpowspec = meas.ResSpecPower(cfg_dict, qi=0, params={'rng': 300, 'span': 5, 'f_off': 1, 'expts_gain': 30})

# Power sweep with qubit in excited state
rpowspec_e = meas.ResSpecPower(cfg_dict, qi=0, params={'pulse_e': True, 'span': 5})
```

The experiment produces a 2D plot showing the resonator response as a function of both frequency and power. The Lamb shift (difference in resonator frequency between high and low power) can be extracted from the fit.

#### Qubit Spectroscopy

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Measures the qubit frequency by applying a probe pulse with variable frequency and measuring the resulting qubit state. This allows determination of the qubit transition frequencies (f_ge and f_ef).

The module includes:
- `QubitSpecProgram`: Defines the pulse sequence for the spectroscopy experiment
- `QubitSpec`: Main experiment class for frequency spectroscopy
- `QubitSpecPower`: 2D version that sweeps both frequency and power

This experiment is particularly useful for finding qubit frequencies and characterizing the qubit spectrum as a function of probe power.

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `gain`: Probe pulse amplitude
- `length`: Probe pulse length (μs) (can be set to "t1" to use 3*T1 for GE or T1/4 for EF)
- `checkEF`: Whether to check the e-f transition
- `pulse_type`: Type of pulse ('const' or 'gauss')
- `sep_readout`: Whether to separate the probe pulse and readout
- `readout_length`: Length of the readout pulse
- `style`: 'huge', 'coarse', 'medium', or 'fine' for different scan ranges and powers

**Example**:
```python
# Find g-e transition with medium range
qspec = meas.QubitSpec(cfg_dict, qi=0, style='medium')

# Find e-f transition
qspec_ef = meas.QubitSpec(cfg_dict, qi=0, style='medium', params={'checkEF': True})

# Fine scan with low power
qspec_fine = meas.QubitSpec(cfg_dict, qi=0, style='fine')
```

#### Qubit Spectroscopy Power

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Performs a 2D sweep of frequency and power to map out the qubit response. This experiment is useful for identifying power-dependent frequency shifts, multi-photon transitions, and optimal drive powers. It creates a 2D map showing how the qubit spectrum changes with probe power.

**Key Parameters**:
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `reps`: Number of repetitions for each experiment
- `rng`: Range for logarithmic gain sweep (default: 50)
- `max_gain`: Maximum gain value (default: qubit.max_gain)
- `expts_gain`: Number of gain points (default: 10)
- `log`: Whether to use logarithmic gain spacing (default: True)
- `checkEF`: Whether to check the e-f transition
- `style`: 'coarse' for wide frequency span with many points, 'fine' for narrow span with fewer points

**Example**:
```python
# Standard power sweep
qspec_pow = meas.QubitSpecPower(cfg_dict, qi=0, params={'span': 40, 'expts': 100, 'max_gain': 0.4})

# Power sweep for EF transition
qspec_pow_ef = meas.QubitSpecPower(cfg_dict, qi=0, params={'checkEF': True, 'span': 20})

# Fine scan with more gain points
qspec_pow_fine = meas.QubitSpecPower(cfg_dict, qi=0, style='fine', params={'expts_gain': 20})
```

The experiment produces a 2D plot showing the qubit response as a function of both frequency and power. This can reveal features like AC Stark shifts, multi-photon transitions, and power broadening.

#### Rabi Experiment

**File**: `experiments/single_qubit/rabi.py`

**Description**: Measures Rabi oscillations by varying either the amplitude or length of a drive pulse. Rabi oscillations are observed by varying either the amplitude or length of a driving pulse and measuring the resulting qubit state. This allows determination of the π-pulse parameters (amplitude and duration) needed for qubit control.

The module includes:
- `RabiProgram`: Defines the pulse sequence for the Rabi experiment
- `RabiExperiment`: Main experiment class for amplitude or length Rabi oscillations
- `ReadoutCheck`: Class for checking readout parameters
- `RabiChevronExperiment`: 2D version that sweeps both frequency and amplitude/length

**Key Parameters**:
- `expts`: Number of amplitude/length points (default: 60)
- `reps`: Number of repetitions for each experiment
- `rounds`: Number of software averages
- `gain`: Maximum pulse amplitude (for length sweep)
- `sigma`: Pulse width (for amplitude sweep)
- `sweep`: 'amp' or 'length' to specify what to sweep
- `checkEF`: Whether to check the e-f transition (default: False)
- `pulse_ge`: Boolean flag to indicate if pulse is for ground to excited state transition (default: True)
- `start`: Starting point for the experiment (default: 0)
- `pulse_type`: Type of pulse used in the experiment (default: 'gauss')

**Example**:
```python
# Amplitude Rabi
amp_rabi = meas.RabiExperiment(cfg_dict, qi=0)

# Length Rabi
len_rabi = meas.RabiExperiment(cfg_dict, qi=0, params={'sweep': 'length', 'type': 'const', 'sigma': 0.1, 'gain': 0.15})

# EF Rabi (checks e-f transition)
ef_rabi = meas.RabiExperiment(cfg_dict, qi=0, params={'checkEF': True})
```

#### Rabi Chevron

**File**: `experiments/single_qubit/rabi.py`

**Description**: Performs a 2D sweep of pulse amplitude/length and frequency to map out Rabi oscillations as a function of detuning. This experiment creates a 2D map showing how the Rabi oscillation frequency changes with detuning from the qubit frequency, creating a characteristic chevron pattern.

**Key Parameters**:
- `span_f`: Frequency span (MHz)
- `expts_f`: Number of frequency points (default: 30)
- `sweep`: 'amp' or 'length' to specify what to sweep (default: 'amp')
- `checkEF`: Whether to check the e-f transition (default: False)
- `start_f`: Start qubit frequency (MHz) (default: f_ge - span_f/2)
- `type`: Type of sweep ('amp' or 'length')

**Example**:
```python
# Amplitude Rabi Chevron
amp_rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=0, params={'span_f': 10})

# Length Rabi Chevron
len_rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=0, params={'span_f': 10, 'sweep': 'length'})
```

The experiment produces a 2D plot showing the qubit response as a function of both drive frequency and amplitude/length. The chevron pattern allows visualization of how the Rabi oscillation frequency increases with detuning from the qubit frequency.

#### T1 Experiment

**File**: `experiments/single_qubit/t1.py`

**Description**: Measures the energy relaxation time (T1) by exciting the qubit to the |1⟩ state and measuring its decay over time. T1 (energy relaxation time) is measured by:
1. Exciting the qubit to |1⟩ state with a π pulse
2. Waiting for a variable delay time
3. Measuring the qubit state
4. Fitting the decay of the |1⟩ state population to an exponential function

The module provides several experiment classes:
- `T1Program`: Low-level pulse sequence implementation
- `T1Experiment`: Standard T1 measurement
- `T1_2D`: 2D T1 measurement for stability analysis

**Key Parameters**:
- `expts`: Number of wait time points (default: 60)
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rounds`: Number of software averages
- `start`: Start wait time (μs) (default: 0)
- `span`: Total span of wait times (μs) (default: 3.7 * T1)
- `acStark`: Whether to apply AC Stark shift during wait time (default: False)
- `active_reset`: Whether to use active qubit reset
- `qubit`: List of qubit indices to measure

**Example**:
```python
# Standard T1 measurement
t1 = meas.T1Experiment(cfg_dict, qi=0)

# T1 with AC Stark shift
t1_stark = meas.T1Experiment(cfg_dict, qi=0, params={'acStark': True, 'stark_freq': 4900, 'stark_gain': 0.1})

# 2D T1 measurement for stability analysis
t1_2d = meas.T1_2D(cfg_dict, qi=0, params={'sweep_pts': 100})
```

The experiment fits the data to an exponential decay function and extracts the T1 time. The result can be automatically updated in the configuration file using the `update()` method.

#### T2 Ramsey Experiment

**File**: `experiments/single_qubit/t2.py`

**Description**: Measures the phase coherence time (T2*) using a Ramsey sequence (π/2 - wait - π/2). T2 is a measure of how long a qubit maintains phase coherence in the x-y plane of the Bloch sphere. The Ramsey protocol uses two π/2 pulses separated by a variable delay time.

The module supports two main measurement protocols:
1. Ramsey: Uses two π/2 pulses separated by a variable delay time
2. Echo: Uses two π/2 pulses with one or more π pulses in between to refocus dephasing

Additional features include:
- AC Stark shift measurements during Ramsey experiments
- EF transition measurements (first excited to second excited state)
- Automatic frequency error detection and correction

**Key Parameters**:
- `experiment_type`: "ramsey" or "echo" (default: "ramsey")
- `expts`: Number of wait time points (default: 100)
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rounds`: Number of software averages
- `start`: Start wait time (μs) (default: 0.01)
- `span`: Total span of wait times (μs) (default: 3 * T2r)
- `ramsey_freq`: Frequency detuning for phase advancement (MHz) (default: "smart", which sets to 1.5/T2)
- `checkEF`: Whether to check the e-f transition (default: False)
- `acStark`: Whether to apply AC Stark shift during wait time (default: False)
- `active_reset`: Whether to use active qubit reset

**Example**:
```python
# Standard Ramsey experiment
t2r = meas.T2Experiment(cfg_dict, qi=0)

# Ramsey with specific detuning frequency
t2r_custom = meas.T2Experiment(cfg_dict, qi=0, params={'ramsey_freq': 0.5})

# Ramsey for EF transition
t2r_ef = meas.T2Experiment(cfg_dict, qi=0, params={'checkEF': True})
```

The experiment fits the data to a decaying sinusoid and extracts both the T2 time and the frequency error. The frequency error can be used to correct the qubit frequency in the configuration.

#### T2 Echo Experiment

**File**: `experiments/single_qubit/t2.py`

**Description**: Measures the echo coherence time (T2E) using a Hahn echo sequence (π/2 - wait/2 - π - wait/2 - π/2). The echo protocol adds one or more π pulses between the π/2 pulses to refocus dephasing caused by low-frequency noise, typically resulting in longer coherence times than the Ramsey measurement.

**Key Parameters**:
- `experiment_type`: Set to 'echo' for echo experiment
- `expts`: Number of wait time points (default: 100)
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rounds`: Number of software averages
- `start`: Start wait time (μs) (default: 0.01)
- `span`: Total span of wait times (μs) (default: 3 * T2e)
- `num_pi`: Number of π pulses (default: 1 for standard echo)
- `active_reset`: Whether to use active qubit reset

**Example**:
```python
# Standard Echo experiment
t2e = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo'})

# Echo with multiple π pulses (CPMG sequence)
t2e_cpmg = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo', 'num_pi': 3})
```

The experiment fits the data to a decaying sinusoid (or exponential if no oscillations are visible) and extracts the T2E time. The T2E time is typically longer than the T2* time measured by the Ramsey experiment because the echo sequence refocuses dephasing caused by low-frequency noise.

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

#### Active Reset

**File**: `experiments/single_qubit/active_reset.py`

**Description**: This experiment does not do any fitting. It is designed to test the memory and the active reset.

**Key Parameters**:
- `shots`: Number of shots per experiment
- `check_e`: Whether to test the e state blob (true if unspecified)
- `check_f`: Whether to also test the f state blob
- `active_reset`: Boolean to add active reset
- `read_wait`: Wait time between measurements in us

**Example**:
```python
reset_exp = meas.RepMeasExperiment(cfg_dict, qi=0, params={'shots': 20000, 'active_reset': True})
```

#### Single Shot Optimization

**File**: `experiments/single_qubit/single_shot_opt.py`

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

#### T1 Continuous Measurement

**File**: `experiments/single_qubit/t1_cont.py`

**Description**: This experiment performs continuous T1 measurements to monitor the stability of the qubit's relaxation time. It repeatedly runs a T1 experiment and plots the results over time.

**Key Parameters**:
- `t1_expts`: Number of T1 experiments to run
- `t1_reps`: Number of repetitions for each T1 experiment
- `t1_rounds`: Number of software averages for each T1 experiment

**Example**:
```python
t1_cont = meas.T1Cont(cfg_dict, qi=0, params={'t1_expts': 100})
```

#### T1 Stark Measurement

**File**: `experiments/single_qubit/t1_stark.py`

**Description**: This experiment measures the T1 relaxation time in the presence of a Stark drive applied to the readout resonator. It is useful for characterizing the impact of the Stark drive on qubit coherence.

**Key Parameters**:
- `stark_gain`: Amplitude of the Stark pulse
- `stark_freq`: Frequency of the Stark pulse
- `wait_time`: Wait time for the T1 measurement

**Example**:
```python
t1_stark = meas.T1Stark(cfg_dict, qi=0, params={'stark_gain': 0.1, 'stark_freq': 4900})
```

#### T2 Ramsey Stark Measurement

**File**: `experiments/single_qubit/t2_ramsey_stark.py`

**Description**: This experiment measures the T2 Ramsey coherence time while applying a Stark drive to the readout resonator. It helps to understand how the Stark drive affects qubit dephasing.

**Key Parameters**:
- `stark_gain`: Amplitude of the Stark pulse
- `stark_freq`: Frequency of the Stark pulse
- `ramsey_freq`: Ramsey frequency for phase advancement

**Example**:
```python
t2_ramsey_stark = meas.T2RamseyStark(cfg_dict, qi=0, params={'stark_gain': 0.1, 'stark_freq': 4900})
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

#### Two-Qubit T1 Continuous

**File**: `experiments/two_qubit/t1_2q_cont.py`

**Description**: This experiment continuously measures the T1 relaxation time for two qubits simultaneously. It is useful for monitoring the stability of T1 for both qubits over time.

**Key Parameters**:
- `shots`: Number of shots per experiment
- `reps`: Number of repetitions for each experiment
- `rounds`: Number of software averages
- `wait_time`: Wait time for the T1 measurement
- `active_reset`: Whether to use active reset
- `final_delay`: Delay between measurements
- `readout`: Readout pulse length
- `n_g`: Number of ground state measurements
- `n_e`: Number of excited state measurements
- `n_t1`: Number of T1 measurements

**Example**:
```python
t1_2q_cont = meas.T1Cont2QExperiment(cfg_dict, qi=[0, 1], params={'shots': 50000})
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
| `rounds` | Number of software averages | Varies by experiment |
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
