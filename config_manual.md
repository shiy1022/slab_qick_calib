# Manual for `config.py`

This manual explains the parameters and functions in the `config.py` module and how they are used in various quantum experiments.

## Overview

The `config.py` module provides a set of functions to initialize, load, save, and update configuration files for quantum experiments with superconducting qubits. It creates YAML configuration files with parameters for qubits, readout resonators, and hardware settings. This configuration is used by various experiments in the `experiments` folder.

## Configuration Management Functions

### `load(file_name)`
Loads a YAML configuration file and returns it as an `AttrDict`, which allows accessing dictionary keys as attributes.

### `save(cfg, file_name, reload=True)`
Saves a configuration object to a YAML file. If `reload` is `True`, it reloads the file after saving and returns the updated configuration.

### `save_copy(file_name)`
Saves a copy of a configuration file with a timestamp in the filename to prevent overwriting previous configurations.

### `update_config(file_name, path, field, value, index=None, ...)`
A general-purpose function to update a specific value in the configuration file. It can update any part of the configuration, including nested fields and array elements.

**Update Helper Functions:**
- `update_qubit(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update qubit parameters.
- `update_readout(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update readout parameters.
- `update_stark(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update Stark shift parameters.
- `update_lo(file_name, field, value, qi, ...)`: A wrapper for `update_config` to update local oscillator parameters.

### `save_single_qubit_config(file_name, qubit_index, new_file_name)`
Extracts the configuration for a single qubit and saves it to a new file. This is useful for running single-qubit experiments without loading the full multi-qubit configuration.

## Configuration Initialization Functions

### `init_config(file_name, num_qubits, type="full", t1=50, aliases="Qick001")`
Initializes a standard configuration file for multi-qubit experiments.

| Parameter | Description |
|-----------|-------------|
| `file_name` | Path to the output YAML configuration file. |
| `num_qubits`| Number of qubits to configure. |
| `type` | Type of readout DAC output (`full`, `mux`, `int`). |
| `t1` | Default T1 relaxation time in μs, used to set initial T1, T2, and readout delay values. |
| `aliases` | Identifier for the System-on-Chip (SoC). |

### `init_config_res(file_name, num_qubits, type="full", aliases="Qick001")`
Initializes a configuration file specifically for resonator experiments. This configuration is a simplified version of the standard config, containing only the parameters necessary for resonator characterization.

### `init_model_config(file_name, num_qubits)`
Initializes a model configuration file for storing theoretical and fitting parameters related to the quantum device. This is used for more advanced analysis and modeling.

## Configuration Structure

The configuration is organized into three main sections:

1. `device.qubit`: Qubit parameters
2. `device.readout`: Readout resonator parameters
3. `hw.soc`: Hardware configuration

## Qubit Parameters (`device.qubit`)

### Coherence Times

| Parameter | Description |
|-----------|-------------|
| `T1` | Energy relaxation time (μs). |
| `T2r` | Ramsey dephasing time (μs). |
| `T2e` | Echo dephasing time (μs). |

### Qubit Frequencies

| Parameter | Description |
|-----------|-------------|
| `f_ge` | Ground to excited state transition frequency (MHz). |
| `f_ef` | Excited to second excited state transition frequency (MHz). |
| `kappa` | Qubit linewidth (MHz) (informational). |

### Qubit Pulses (`device.qubit.pulses`)

| Parameter | Description |
|-----------|-------------|
| `pi_ge.gain`| Amplitude of the π pulse for the g→e transition. |
| `pi_ge.sigma`| Width (std dev) of the Gaussian π pulse for g→e. |
| `pi_ef.gain`| Amplitude of the π pulse for the e→f transition. |
| `pi_ef.sigma`| Width (std dev) of the Gaussian π pulse for e→f. |
| `*.sigma_inc`| Pulse length in units of sigma. |
| `*.type` | Pulse shape type (e.g., "gauss"). |

### Other Qubit Parameters

| Parameter | Description |
|-----------|-------------|
| `spec_gain` | Gain scaling factor for spectroscopy. |
| `pop` | Thermal population (informational). |
| `temp` | Qubit temperature (informational). |
| `tuned_up` | Flag indicating if the qubit is tuned up. |
| `low_gain` | Minimum gain for spectroscopy scans. |
| `max_gain` | Maximum gain for qubit control pulses. |

## Readout Parameters (`device.readout`)

### Resonator Properties

| Parameter | Description |
|-----------|-------------|
| `frequency` | Readout resonator frequency (MHz). |
| `gain` | Readout pulse amplitude. |
| `lamb` | Lamb shift (informational). |
| `chi` | Dispersive shift (informational). |
| `kappa` | Resonator linewidth (MHz). |
| `qe` | External quality factor (informational). |
| `qi` | Internal quality factor (informational). |

### Readout Settings

| Parameter | Description |
|-----------|-------------|
| `phase` | Phase rotation for the readout signal. |
| `readout_length`| Duration of the readout pulse (μs). |
| `threshold` | State discrimination threshold. |
| `fidelity` | Readout fidelity (informational). |
| `tm` | Measurement time / T1 fit from single shot (informational). |
| `sigma` | Width (noise) of the readout histogram (informational). |
| `rescale` | Flag to rescale readout data. |

### Readout Timing

| Parameter | Description |
|-----------|-------------|
| `trig_offset`| Trigger offset for readout. |
| `final_delay`| Delay after readout before the next experiment (μs), typically set to `6 * T1`. |
| `active_reset`| Flag to enable active qubit reset. |
| `reset_e` | Parameter for active reset of the excited state. |
| `reset_g` | Parameter for active reset of the ground state. |

### Readout Averaging

| Parameter | Description |
|-----------|-------------|
| `reps` | Per-qubit multiplier for the number of repetitions. |
| `rounds` | Per-qubit multiplier for the number of software averages. |
| `reps_base` | Base number of repetitions for the entire device. |
| `rounds_base`| Base number of software averages for the entire device. |
| `max_gain` | Maximum gain for the readout pulse. |

## Hardware Configuration (`hw.soc`)

### ADC Configuration

| Parameter | Description |
|-----------|-------------|
| `adcs.readout.ch`| ADC channel for readout. |
| `adcs.readout.type`| ADC type (e.g., "dyn"). |

### DAC Configuration

| Parameter | Description |
|-----------|-------------|
| `dacs.qubit.ch` | DAC channel for qubit control. |
| `dacs.qubit.nyquist`| Nyquist zone for the qubit DAC. |
| `dacs.qubit.type` | Type of qubit DAC output (`full`, `mux`, `int`). |
| `dacs.readout.ch`| DAC channel for readout. |
| `dacs.readout.nyquist`| Nyquist zone for the readout DAC. |
| `dacs.readout.type` | Type of readout DAC output (`full`, `mux`, `int`). |

## Conclusion

The `config.py` module provides a flexible and comprehensive system for managing configurations in quantum experiments. Understanding these parameters and functions is essential for effectively running, customizing, and analyzing the results of quantum experiments with this codebase.
