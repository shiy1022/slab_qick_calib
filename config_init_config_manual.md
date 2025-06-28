# Manual for `config.init_config`

This manual explains the parameters in the `init_config` function from `config.py` and how they are used in various quantum experiments.

## Overview

The `init_config` function initializes a configuration file for quantum experiments with superconducting qubits. It creates a YAML configuration file with parameters for qubits, readout resonators, and hardware settings. This configuration is used by various experiments in the `experiments` folder.

```python
def init_config(file_name, num_qubits, type="full", t1=50, aliases="Qick001"):
    # Creates a configuration file with default parameters for num_qubits qubits
```

## Parameters

### Function Parameters

| Parameter | Description | Usage |
|-----------|-------------|-------|
| `file_name` | Path to the output YAML configuration file | Used to save the configuration |
| `num_qubits` | Number of qubits to configure | Determines the length of parameter arrays |
| `type` | Type of readout, default is "full" | Used in hardware configuration for readout DACs |
| `t1` | Default T1 relaxation time in μs | Used to initialize qubit T1, T2 values and readout delay |
| `aliases` | Identifier for the System-on-Chip (SoC) | Used in hardware configuration |

## Configuration Structure

The configuration is organized into three main sections:

1. `device.qubit`: Qubit parameters
2. `device.readout`: Readout resonator parameters
3. `hw.soc`: Hardware configuration

## Qubit Parameters (`device.qubit`)

### Coherence Times

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `T1` | Energy relaxation time (μs) | Used in `t1.py` to set the span of wait times for T1 measurements. Also used in `rabi.py` when calculating pulse lengths. |
| `T2r` | Ramsey dephasing time (μs) | Used in T2 Ramsey experiments to set appropriate measurement timescales. |
| `T2e` | Echo dephasing time (μs) | Used in T2 Echo experiments to set appropriate measurement timescales. |

### Qubit Frequencies

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `f_ge` | Ground to excited state transition frequency (MHz) | Used in `pulse_probe_spectroscopy.py` to set the center frequency for spectroscopy. Used in `rabi.py` to set the frequency of the π pulse. |
| `f_ef` | Excited to second excited state transition frequency (MHz) | Used in `pulse_probe_spectroscopy.py` when `checkEF=True` to probe the ef transition. |
| `kappa` | Qubit linewidth (MHz) | Used in spectroscopy experiments to determine appropriate frequency spans. |

### Qubit Pulses

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `pulses.pi_ge.gain` | Amplitude of π pulse for g→e transition | Used in `rabi.py` and other experiments that apply π pulses to the qubit. |
| `pulses.pi_ge.sigma` | Width parameter for Gaussian π pulse (g→e) | Used in `rabi.py` to set the pulse width. |
| `pulses.pi_ge.sigma_inc` | Number of sigma at which pulse is cropped | Used to calculate the total pulse length from sigma. |
| `pulses.pi_ge.type` | Pulse shape type (e.g., "gauss") | Determines the pulse shape in experiments. |
| `pulses.pi_ef.gain` | Amplitude of π pulse for e→f transition | Used when manipulating the second excited state. |
| `pulses.pi_ef.sigma` | Width parameter for Gaussian π pulse (e→f) | Used to set the pulse width for e→f transitions. |
| `pulses.pi_ef.sigma_inc` | Number of sigma at which pulse is cropped | Used to calculate the total pulse length from sigma. |
| `pulses.pi_ef.type` | Pulse shape type (e.g., "gauss") | Determines the pulse shape in experiments. |

### Other Qubit Parameters

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `spec_gain` | Gain scaling factor for spectroscopy for qubit to qubit variation | Used in `pulse_probe_spectroscopy.py` to set appropriate pulse amplitudes. |
| `pop` | Thermal population | Used in analysis of qubit measurements. |
| `temp` | Qubit temperature | Used in analysis of qubit measurements. |
| `tuned_up` | Boolean flag indicating if qubit is tuned | Used in experiments to determine whether to show additional diagnostic information. |
| `low_gain` | Gain for finest spectroscopy scan | Used in `pulse_probe_spectroscopy.py` and `rabi.py` to set the minimum gain for pulses. |
| `max_gain` | Maximum gain value for pulses (RFSoC property) | Used in `pulse_probe_spectroscopy.py` and `rabi.py` to set the maximum gain for pulses. |

## Readout Parameters (`device.readout`)

### Resonator Properties

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `frequency` | Readout resonator frequency (MHz) | Used in `resonator_spectroscopy.py` to set the center frequency for spectroscopy. |
| `gain` | Readout pulse amplitude | Used in all experiments that perform qubit readout. |
| `lamb` | Lamb shift | Used in analysis of resonator spectroscopy data. |
| `chi` | Dispersive shift | Used in analysis of resonator spectroscopy data. |
| `kappa` | Resonator linewidth (MHz) | Used in `resonator_spectroscopy.py` to determine appropriate frequency spans. |
| `qe` | External quality factor in units of 10k | Used in analysis of resonator spectroscopy data. |
| `qi` | Internal quality factor in units of 10k | Used in analysis of resonator spectroscopy data. |

### Readout Settings

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `phase` | Phase rotation for readout signal | Used in all experiments to correctly process the readout signal. |
| `readout_length` | Duration of readout pulse (μs) | Used in all experiments that perform qubit readout. |
| `threshold` | Threshold for state discrimination | Used for active reset. |
| `fidelity` | Readout fidelity | Used in analysis of readout performance. |
| `tm` | Time constant for readout | Used in analysis of readout performance. |
| `sigma` | Width parameter for readout histogram | Used in alnalysis of readout performance. |
| `trig_offset` | Trigger offset for readout | Used in timing of readout pulses. |
| `final_delay` | Delay after readout before next experiment (μs) | Used in all experiments to ensure qubit returns to ground state. Set to 6*T1 by default. |
| `active_reset` | Boolean flag for active qubit reset | Used in experiments that support active reset to improve experiment speed. |
| `reset_e` | Parameter for active reset of excited state | Used in active reset protocols. |
| `reset_g` | Parameter for active reset of ground state | Used in active reset protocols. |
| `reps` | If qubit needs more or fewer reps than reps_base, set param here (1 standard) | Used in all experiments to set the number of repetitions. |
| `soft_avgs` | If qubit needs more or fewer soft_avgs than soft_avgs_base, set param here (1 standard) | Used in all experiments to set the number of software averages. |
| `reps_base` | Base number of repetitions for entire device | Used to calculate appropriate repetition counts. |
| `soft_avgs_base` | Base number of software averages for entire device | Used to calculate appropriate averaging counts. |
| `max_gain` | Maximum gain for readout, RFSoC parameter (usually 1) | Used to limit readout pulse amplitude. |

## Hardware Configuration (`hw.soc`)

### ADC Configuration

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `adcs.readout.ch` | ADC channel for readout | Used in all experiments to specify which ADC channel to use for readout. |

### DAC Configuration

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `dacs.qubit.ch` | DAC channel for qubit control | Used in all experiments to specify which DAC channel to use for qubit control. |
| `dacs.qubit.nyquist` | Nyquist zone for qubit DAC, 1 for frequencies < fs, 2 for frequencies above | Used in signal generation for qubit control. |
| `dacs.qubit.type` | Type of qubit DAC output, full/mux/int but only full supported now | Used in signal generation for qubit control. |
| `dacs.readout.ch` | DAC channel for readout | Used in all experiments to specify which DAC channel to use for readout. |
| `dacs.readout.nyquist` | Nyquist zone for readout DAC, 1 for frequencies < fs, 2 for frequencies above | Used in signal generation for readout. |
| `dacs.readout.type` | Type of readout DAC output, full/mux/int but only full supported now | Used in signal generation for readout. |

## Usage Examples

### Resonator Spectroscopy

In `resonator_spectroscopy.py`, the resonator frequency is swept to find the resonance:

```python
# From ResSpec class in resonator_spectroscopy.py
params_def["center"] = self.cfg.device.readout.frequency[qi]
params_def["expts"] = 220
params_def["span"] = 5
```

The experiment uses the configured resonator frequency as the center for the frequency sweep.

### Qubit Spectroscopy

In `pulse_probe_spectroscopy.py`, the qubit frequency is swept to find the qubit transition:

```python
# From QubitSpec class in pulse_probe_spectroscopy.py
if params['checkEF']:
    params_def["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
else:
    params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
```

The experiment uses either the g→e or e→f transition frequency as the center for the frequency sweep.

### Rabi Oscillations

In `rabi.py`, the amplitude or length of a qubit drive pulse is swept to observe Rabi oscillations:

```python
# From RabiExperiment class in rabi.py
if params['checkEF']:
    cfg_qub = self.cfg.device.qubit.pulses.pi_ef
    params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
else:
    cfg_qub = self.cfg.device.qubit.pulses.pi_ge
    params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
```

The experiment uses the configured qubit frequency and pulse parameters to generate the appropriate drive pulse.

### T1 Measurement

In `t1.py`, the delay after a π pulse is swept to measure the energy relaxation time:

```python
# From T1Experiment class in t1.py
params_def = {
    # 
    "span": 3.7 * self.cfg.device.qubit.T1[qi],  # Total span of wait times (μs), set to ~3.7*T1
    # 
}
```

The experiment uses the configured T1 time to set an appropriate span for the delay sweep.

## Updating Configuration

The `config.py` module provides functions to update the configuration based on experiment results:

```python
# From T1Experiment.update method in t1.py
def update(self, cfg_file, rng_vals=[1,500], first_time=False, verbose=True):
    qi = self.cfg.expt.qubit[0]
    if self.status: 
        config.update_qubit(cfg_file, 'T1', self.data['new_t1_i'], qi, sig=2, rng_vals=rng_vals, verbose=verbose)
        config.update_readout(cfg_file, 'final_delay', 6*self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0]*10, rng_vals[1]*3], verbose=verbose)
        if first_time:
            config.update_qubit(cfg_file, 'T2r', self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0], rng_vals[1]*2], verbose=verbose)        
            config.update_qubit(cfg_file, 'T2e', 2*self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0], rng_vals[1]*2], verbose=verbose)
```

This allows experiments to automatically update the configuration with measured values, creating a feedback loop for qubit tuning.

## Conclusion

The `init_config` function provides a comprehensive set of default parameters for quantum experiments with superconducting qubits. These parameters are used throughout the experiments in the `experiments` folder to configure pulse sequences, measurement settings, and hardware interfaces. Understanding these parameters is essential for effectively running and customizing quantum experiments with this codebase.
