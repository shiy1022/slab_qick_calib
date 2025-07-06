# Superconducting Qubit Tuning Guide

This guide provides a comprehensive overview of how to tune superconducting qubits using the `tune_qubits_basic2` notebook and the underlying calibration framework.

## Table of Contents
1. [Introduction](#introduction)
2. [The Complete Tuning Workflow](#the-complete-tuning-workflow)
3. [Understanding qubit_list and update Mechanism](#understanding-qubit_list-and-update-mechanism)
4. [Measurement Types and Fitting Procedures](#measurement-types-and-fitting-procedures)
5. [Configuration Parameters Reference](#configuration-parameters-reference)
6. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)

## Introduction

Superconducting qubit tuning is the process of characterizing and optimizing the control parameters for quantum operations. This involves measuring fundamental qubit properties like coherence times, transition frequencies, and optimal pulse parameters, then updating the configuration file with these values for future experiments.

### Key Concepts
- **Configuration File**: YAML file storing all qubit and readout parameters
- **Fitting**: Mathematical analysis to extract physical parameters from measurement data
- **Update Mechanism**: Automatic updating of configuration parameters based on measurement results
- **Quality Metrics**: R² values and fit errors to validate measurement quality

## The Complete Tuning Workflow

The tuning process follows a systematic sequence designed to progressively characterize and optimize qubit performance:

### 1. Initial Setup and Connection
```python
# Connect to RFSoC and load configuration
cfg_dict = {'soc': soc, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}
```

### 2. Time of Flight (TOF) Calibration
**Purpose**: Determine the signal propagation delay through cables
```python
tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=qi)
```
**Updates**: `trig_offset` parameter for proper measurement timing

### 3. Resonator Characterization

#### Coarse Resonator Spectroscopy
**Purpose**: Find all resonators on the feedline
```python
rspecc = meas.ResSpec(cfg_dict, qi=qi, style='coarse', progress=True, 
                      params={'start':6000, 'span':2000, 'reps':1000, 'gain':0.01, 'expts':6000})
```
**Fitting**: Peak finding algorithm to identify resonator frequencies
**Updates**: Initial frequency estimates for each resonator

#### Fine Resonator Spectroscopy
**Purpose**: Precisely characterize individual resonators
```python
rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':5, 'center':res_values[qi]})
if update: rspec.update(cfg_dict['cfg_file'])
```
**Fitting**: Hanger function fit: `hangerS21func_sloped(f, f0, Qi, Qe, phi, scale, slope)`
**Updates**:
- `frequency`: Resonator frequency (f0)
- `kappa`: Resonator linewidth = f0 * (1/Qi + 1/Qe)
- `qi`: Internal quality factor
- `qe`: External quality factor

#### Resonator Power Spectroscopy
**Purpose**: Find optimal readout power and measure Lamb shift
```python
rpowspec = meas.ResSpecPower(cfg_dict, qi=qi, params={'rng':300,'max_gain':1, 'span':18,"f_off":4,'expts_gain':30})
if update:
    auto_cfg = config.update_readout(cfg_path, 'lamb', rpowspec.data['lamb_shift'], qi)
```
**Fitting**: Hanger fits at high and low power
**Updates**:
- `lamb`: Lamb shift (frequency difference between high and low power)
- `gain`: Optimal readout gain

### 4. Qubit Spectroscopy

#### Automated Qubit Finding
```python
status, ntries = qubit_tuning.find_spec(qi, cfg_dict, start='coarse')
```
**Purpose**: Locate qubit transition frequency with iterative refinement of gain/scan width.
**Fitting**: Lorentzian fit: `lorfunc(f, y0, yscale, f0, linewidth)`
**Updates**:
- `f_ge`: Qubit |g⟩→|e⟩ transition frequency
- `kappa`: Qubit linewidth (2 × fit linewidth)

#### Manual Qubit Spectroscopy
```python
qspec = meas.QubitSpec(cfg_dict, qi=qi, style='medium')
if update and qspec.status: 
    auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data["best_fit"][2], qi)
```
**Styles Available**:
- `'huge'`: Very wide scan (±750 MHz)
- `'coarse'`: Wide scan (±250 MHz) 
- `'medium'`: Medium scan (±25 MHz)
- `'fine'`: Narrow scan (±2.5 MHz)

### 5. Pulse Calibration

#### Amplitude Rabi Oscillations
**Purpose**: Calibrate π-pulse amplitude
```python
amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'reps':20000})
if update and amp_rabi.status:
    config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)
```
**Fitting**: Sinusoidal fit: `sinfunc(x, yscale, freq, phase_deg, y0)`
**Analysis**: π-pulse gain = `fix_phase(fit_params)` 
**Updates**: `pulses.pi_ge.gain` - amplitude for π rotation

#### Length Rabi Oscillations
**Purpose**: Calibrate π-pulse duration (for constant pulses)
```python
len_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'sweep':'length', 'pulse_type':'const'})
```
**Updates**: `pulses.pi_ge.sigma` or `pulses.pi_ge.length` depending on pulse type

### 6. Coherence Time Measurements

#### T1 (Energy Relaxation Time)
**Purpose**: Measure qubit energy decay time
```python
t1 = meas.T1Experiment(cfg_dict, qi=qi, params={'reps':100000,'span':0, 'start':20})
if update: t1.update(cfg_path, first_time=first_time)
```
**Sequence**: π pulse → variable delay → measurement
**Fitting**: Exponential decay: `expfunc(t, y0, yscale, T1)`
**Updates**:
- `T1`: Energy relaxation time
- `final_delay`: Wait time between experiments (6 × T1)
- `T2r`, `T2e`: Initial estimates if first_time=True

#### T2 Ramsey (Dephasing Time)
**Purpose**: Measure qubit dephasing time and correct frequency
```python
t2r = meas.T2Experiment(cfg_dict, qi=qi, max_err=10)
if t2r.status and update:
    config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], qi)
    config.update_qubit(cfg_path, 'T2r', t2r.data['best_fit'][3], qi)
```
**Sequence**: π/2 pulse → variable delay → π/2 pulse (with phase)
**Fitting**: Decaying sinusoid: `decayslopesin(t, yscale, freq, phase_deg, T2r, y0, slope)`
**Analysis**: Frequency error = ramsey_freq ± fit_freq (choose smaller error)
**Updates**:
- `T2r`: Ramsey dephasing time
- `f_ge`: Corrected qubit frequency

#### T2 Echo (Coherence Time)
**Purpose**: Measure qubit coherence time with refocusing
```python
t2e = meas.T2Experiment(cfg_dict, qi=qi, params={'experiment_type':'echo'})
if t2e.status and update:
    config.update_qubit(cfg_path, 'T2e', t2e.data['best_fit'][3], qi)
```
**Sequence**: π/2 pulse → delay/2 → π pulse → delay/2 → π/2 pulse
**Fitting**: Decaying sinusoid (same as Ramsey)
**Updates**: `T2e` - echo coherence time

### 7. Readout Optimization

#### Single Shot Histogram
**Purpose**: Measure readout fidelity and optimize threshold
```python
shot = meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':300000})
shot.update(cfg_path)
```
**Analysis**: Gaussian fits to |g⟩ and |e⟩ state distributions
**Updates**:
- `threshold`: Optimal discrimination threshold
- `fidelity`: Readout fidelity
- `sigma`: Readout noise level

#### Readout Parameter Optimization
**Purpose**: Optimize readout gain, frequency, and length
```python
shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=qi, params={'expts_f':1, 'expts_gain':5, 'expts_len':5,'shots':50000})
if update: shotopt.update(cfg_dict['cfg_file'])
```
**Analysis**: 3D optimization over frequency, gain, and length
**Updates**:
- `gain`: Optimal readout gain
- `readout_length`: Optimal readout duration
- `frequency`: Fine-tuned readout frequency

### 8. Advanced Characterization

#### Dispersive Shift (χ) Measurement
**Purpose**: Measure qubit-resonator coupling strength
```python
chi, chi_val = measure_func.check_chi(cfg_dict, qi)
config.update_readout(cfg_path, 'chi', chi_val, qi)
```
**Method**: Compare resonator frequency with qubit in |g⟩ vs |e⟩
**Updates**: `chi` - dispersive shift in MHz

#### EF Transition Characterization
**Purpose**: Characterize |e⟩→|f⟩ transition for two-level control
```python
# Initial frequency estimate
alpha = -150  # Anharmonicity guess
config.update_qubit(cfg_path, 'f_ef', f_ge + alpha, qi)

# EF spectroscopy
qspec = meas.QubitSpec(cfg_dict, qi=qi, style='coarse', params={'checkEF':True})
if update and qspec.status:
    config.update_qubit(cfg_path, 'f_ef', qspec.data["best_fit"][2], qi)

# EF Rabi
amp_rabi = meas.RabiExperiment(cfg_dict, qi=qi, params={'checkEF':True})
if update and amp_rabi.status:
    config.update_qubit(cfg_path, ('pulses','pi_ef','gain'), amp_rabi.data['pi_length'], qi)
```
**Updates**:
- `f_ef`: |e⟩→|f⟩ transition frequency
- `pulses.pi_ef.gain`: π-pulse amplitude for EF transition

## Understanding qubit_list and update Mechanism

### qubit_list Usage
The `qubit_list` parameter allows you to run experiments on multiple qubits:

```python
# Single qubit
qubit_list = [0]

# Multiple qubits
qubit_list = [0, 1, 2]
qubit_list = np.arange(3)  # Same as [0, 1, 2]

# Exclude specific qubits
qubit_list = np.arange(10)
qubit_list = np.delete(qubit_list, [5, 7])  # Remove qubits 5 and 7

# Loop over qubits
for qi in qubit_list:
    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa'})
    if update: rspec.update(cfg_dict['cfg_file'])
```

### update Flag Mechanism
The `update` flag controls whether measurement results are saved to the configuration:

```python
update = True   # Save results to config file
update = False  # Don't save results (analysis only)

# Conditional updating based on fit quality
if update and amp_rabi.status:  # Only update if fit was successful
    config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)
```

### Configuration Update Functions
```python
# Update qubit parameters
config.update_qubit(cfg_path, 'f_ge', new_frequency, qi)
config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), new_gain, qi)

# Update readout parameters  
config.update_readout(cfg_path, 'frequency', new_freq, qi)
config.update_readout(cfg_path, 'gain', new_gain, qi)

# Update with validation (range checking, significance testing)
config.update_qubit(cfg_path, 'T1', new_t1, qi, sig=2, rng_vals=[1, 500])
```

### Automated Tuning
The `tune_up_qubit()` function provides fully automated tuning, generally works better for qubits that have already been set up and need to be re-optimized. 

```python
for qi in qubit_list: 
    qubit_tuning.tune_up_qubit(qi, cfg_dict, first_time=False, single=False, readout=True)
```

**Parameters**:
- `first_time=True`: Assumes no prior calibration, does initial T1 and readout optimization
- `single=True`: Performs single-shot readout optimization
- `readout=True`: Updates readout frequency based on resonator fit

## Measurement Types and Fitting Procedures

### Resonator Spectroscopy
**Experiment**: `ResSpec`
**Fitting Function**: `hangerS21func_sloped(f, f0, Qi, Qe, phi, scale, slope)`
**Physical Model**: Transmission through a hanger-type resonator
**Fit Parameters**:
- `f0`: Resonator frequency
- `Qi`: Internal quality factor  
- `Qe`: External quality factor
- `phi`: Phase offset
- `scale`: Amplitude scaling
- `slope`: Background slope

**Derived Parameters**:
- `kappa = f0 * (1/Qi + 1/Qe)`: Resonator linewidth
- `Q0 = 1/(1/Qi + 1/Qe)`: Total quality factor

**Config Updates**:
- `device.readout.frequency[qi] = f0`
- `device.readout.kappa[qi] = kappa`
- `device.readout.qi[qi] = Qi`
- `device.readout.qe[qi] = Qe`

### Qubit Spectroscopy  
**Experiment**: `QubitSpec`
**Fitting Function**: `lorfunc(f, y0, yscale, f0, linewidth)`
**Physical Model**: Lorentzian absorption line
**Fit Parameters**:
- `y0`: Background level
- `yscale`: Peak amplitude
- `f0`: Qubit frequency
- `linewidth`: Transition linewidth

**Config Updates**:
- `device.qubit.f_ge[qi] = f0`
- `device.qubit.kappa[qi] = 2 * linewidth`

### Rabi Oscillations
**Experiment**: `RabiExperiment`
**Fitting Function**: `sinfunc(x, yscale, freq, phase_deg, y0)`
**Physical Model**: Sinusoidal oscillation of qubit state
**Fit Parameters**:
- `yscale`: Oscillation amplitude
- `freq`: Rabi frequency (MHz)
- `phase_deg`: Phase offset (degrees)
- `y0`: Background offset

**Analysis**: π-pulse gain calculated using `fix_phase()` function:
```python
if phase_deg < 0:
    pi_gain = (1/2 - phase_deg/180) / 2 / freq
else:
    pi_gain = (3/2 - phase_deg/180) / 2 / freq
```

**Config Updates**:
- `device.qubit.pulses.pi_ge.gain[qi] = pi_gain`

### T1 Measurement
**Experiment**: `T1Experiment`  
**Fitting Function**: `expfunc(t, y0, yscale, T1)`
**Physical Model**: Exponential energy relaxation
**Fit Parameters**:
- `y0`: Final population (should be ~0)
- `yscale`: Initial excited state population
- `T1`: Energy relaxation time

**Config Updates**:
- `device.qubit.T1[qi] = T1`
- `device.readout.final_delay[qi] = 6 * T1`

### T2 Ramsey Measurement
**Experiment**: `T2Experiment` (experiment_type='ramsey')
**Fitting Function**: `decayslopesin(t, yscale, freq, phase_deg, T2r, y0, slope)`
**Physical Model**: Decaying sinusoidal oscillation with detuning
**Fit Parameters**:
- `yscale`: Oscillation amplitude
- `freq`: Detuning frequency (MHz)
- `phase_deg`: Phase offset
- `T2r`: Ramsey dephasing time
- `y0`: Background level
- `slope`: Linear drift

**Frequency Correction**:
```python
# Possible frequency errors
f_err_options = [ramsey_freq - fit_freq, ramsey_freq + fit_freq]
f_err = min(f_err_options, key=abs)  # Choose smaller error
new_freq = current_freq + f_err
```

**Config Updates**:
- `device.qubit.T2r[qi] = T2r`
- `device.qubit.f_ge[qi] = new_freq`

### T2 Echo Measurement
**Experiment**: `T2Experiment` (experiment_type='echo')
**Fitting Function**: Same as Ramsey
**Physical Model**: Decaying oscillation with refocusing π-pulse
**Config Updates**:
- `device.qubit.T2e[qi] = T2e`

### Single Shot Readout
**Experiment**: `HistogramExperiment`
**Analysis**: Gaussian fits to |g⟩ and |e⟩ state distributions
**Fit Functions**: Two Gaussian distributions
**Metrics**:
- `fidelity = 1 - (false_positive_rate + false_negative_rate)/2`
- `threshold = (mean_g + mean_e)/2` (optimal discrimination)

**Config Updates**:
- `device.readout.fidelity[qi] = fidelity`
- `device.readout.threshold[qi] = threshold`
- `device.readout.sigma[qi] = sqrt(var_g + var_e)/2`

### Readout Optimization
**Experiment**: `SingleShotOptExperiment`
**Analysis**: 3D grid search over frequency, gain, and length
**Optimization**: Maximize readout fidelity
**Config Updates**:
- `device.readout.gain[qi] = optimal_gain`
- `device.readout.readout_length[qi] = optimal_length`
- `device.readout.frequency[qi] = optimal_frequency`

### Power Spectroscopy
**Experiment**: `ResSpecPower`
**Analysis**: Hanger fits at multiple power levels
**Lamb Shift Calculation**:
```python
lamb_shift = f_high_power - f_low_power
```
**Config Updates**:
- `device.readout.lamb[qi] = lamb_shift`

## Configuration Parameters Reference

### Qubit Parameters (`device.qubit`)

| Parameter | Description | Units | Typical Range | Updated By |
|-----------|-------------|-------|---------------|------------|
| `f_ge[qi]` | |g⟩→|e⟩ transition frequency | MHz | 3000-8000 | QubitSpec, T2Ramsey |
| `f_ef[qi]` | |e⟩→|f⟩ transition frequency | MHz | f_ge - 300 | QubitSpec (checkEF=True) |
| `T1[qi]` | Energy relaxation time | μs | 10-200 | T1Experiment |
| `T2r[qi]` | Ramsey dephasing time | μs | 5-100 | T2Experiment (ramsey) |
| `T2e[qi]` | Echo coherence time | μs | 20-400 | T2Experiment (echo) |
| `kappa[qi]` | Qubit linewidth | MHz | 0.01-1 | QubitSpec |
| `temp[qi]` | Qubit temperature | mK | 10-100 | Temperature measurement |
| `pop[qi]` | Excited state population | - | 0-0.1 | Temperature measurement |
| `tuned_up[qi]` | Tuning status flag | bool | - | Manual/tune_up_qubit |

### Pulse Parameters (`device.qubit.pulses`)

#### π-pulse for |g⟩→|e⟩ (`pi_ge`)
| Parameter | Description | Units | Typical Range | Updated By |
|-----------|-------------|-------|---------------|------------|
| `gain[qi]` | Pulse amplitude | DAC units | 0.1-1.0 | RabiExperiment |
| `sigma[qi]` | Gaussian width | μs | 0.01-0.5 | Manual |
| `sigma_inc[qi]` | Length multiplier | - | 4-6 | Manual |
| `type[qi]` | Pulse shape | - | 'gauss'/'const' | Manual |

#### π-pulse for |e⟩→|f⟩ (`pi_ef`)
| Parameter | Description | Units | Typical Range | Updated By |
|-----------|-------------|-------|---------------|------------|
| `gain[qi]` | Pulse amplitude | DAC units | 0.1-1.0 | RabiExperiment (checkEF=True) |
| `sigma[qi]` | Gaussian width | μs | 0.01-0.5 | Copied from pi_ge |

### Readout Parameters (`device.readout`)

| Parameter | Description | Units | Typical Range | Updated By |
|-----------|-------------|-------|---------------|------------|
| `frequency[qi]` | Readout frequency | MHz | 6000-8000 | ResSpec |
| `gain[qi]` | Readout amplitude | DAC units | 0.001-1.0 | ResSpecPower, SingleShotOpt |
| `readout_length[qi]` | Measurement duration | μs | 1-20 | SingleShotOpt |
| `kappa[qi]` | Resonator linewidth | MHz | 0.1-10 | ResSpec |
| `qi[qi]` | Internal Q factor | - | 1000-100000 | ResSpec |
| `qe[qi]` | External Q factor | - | 100-10000 | ResSpec |
| `chi[qi]` | Dispersive shift | MHz | 0.1-5 | Chi measurement |
| `lamb[qi]` | Lamb shift | MHz | -50 to 50 | ResSpecPower |
| `fidelity[qi]` | Readout fidelity | - | 0.5-0.99 | HistogramExperiment |
| `threshold[qi]` | Discrimination threshold | ADC units | -100 to 100 | HistogramExperiment |
| `sigma[qi]` | Readout noise | ADC units | 1-50 | HistogramExperiment |
| `final_delay[qi]` | Inter-experiment delay | μs | 50-1000 | T1Experiment (6×T1) |
| `trig_offset[qi]` | Measurement timing | μs | 0.2-1.0 | ToFCalibration |
| `active_reset[qi]` | Reset protocol flag | bool | - | Manual |
| `reps[qi]` | Repetitions per point | - | 100-100000 | Manual/adaptive |

### Hardware Parameters (`hw.soc`)

| Parameter | Description | Units | Range | Set By |
|-----------|-------------|-------|-------|--------|
| `adcs.readout.ch[qi]` | ADC channel | - | 0-7 | Manual |
| `dacs.qubit.ch[qi]` | Qubit DAC channel | - | 0-7 | Manual |
| `dacs.readout.ch[qi]` | Readout DAC channel | - | 0-7 | Manual |

### Stark Parameters (`stark`)
Used for AC Stark shift measurements:

| Parameter | Description | Units | Updated By |
|-----------|-------------|-------|------------|
| `q[qi]`, `qneg[qi]` | Quadratic coefficients | MHz/V² | RamseyStarkPower |
| `l[qi]`, `lneg[qi]` | Linear coefficients | MHz/V | RamseyStarkPower |
| `o[qi]`, `oneg[qi]` | Offset coefficients | MHz | RamseyStarkPower |

## Troubleshooting and Best Practices

### Common Issues and Solutions

#### 1. Resonator Spectroscopy Fails
**Symptoms**: No clear resonance peaks, poor fits
**Solutions**:
- Check gain level (too high → punchout, too low → no signal)
- Verify frequency range covers expected resonators
- Adjust prominence parameter in peak finding
- Check cable connections and attenuation

#### 2. Qubit Spectroscopy Fails  
**Symptoms**: No qubit peak visible, fit fails
**Solutions**:
- Increase spectroscopy power (`spec_gain`)
- Expand frequency search range
- Check readout is working (run resonator spec first)
- Verify qubit is in expected frequency range
- Try different pulse lengths

#### 3. Rabi Oscillations Don't Fit
**Symptoms**: No clear oscillations, poor sinusoidal fit
**Solutions**:
- Check qubit frequency is correct
- Adjust gain range (may be too high or too low) or point spacing 
- Increase averaging (`reps`, `rounds`)
- Verify readout fidelity is reasonable (>60%)
- Check for frequency drift

#### 4. T1 Measurement Issues
**Symptoms**: No exponential decay, negative T1
**Solutions**:
- Verify π-pulse is calibrated correctly
- Check for thermal population (measure temperature)
- Increase span if T1 is longer than expected (or decrease for shorter)

#### 5. T2 Ramsey Problems
**Symptoms**: No oscillations, frequency error too large
**Solutions**:
- Adjust Ramsey frequency (try 1/T2r); may have too fast oscillations to resolve, or too slow so that single oscillation looks like decay
- Check π/2 pulse calibration
- Verify qubit frequency is accurate
- Increase span if T2 is longer than expected

#### 6. Poor Readout Fidelity
**Symptoms**: Overlapping |g⟩ and |e⟩ distributions
**Solutions**:
- Optimize readout power (run power sweep)
- Adjust readout frequency (run fine resonator spec)
- Optimize readout length
- Check for mixer calibration issues
- Verify resonator is not saturated

### Best Practices

#### 1. Measurement Order
Always follow this sequence:
1. TOF calibration
2. Resonator spectroscopy (coarse → fine)
3. Readout power optimization
4. Qubit spectroscopy
5. Rabi calibration
6. T1 measurement
7. T2 measurements
8. Readout optimization

#### 2. Parameter Validation
- Always check fit quality (R² > 0.8, low fit errors)
- Validate parameter ranges before updating config
- Use `status` flags to conditionally update
- Monitor parameter drift over time

#### 3. Averaging Strategy
- Start with low averaging for quick scans
- Increase averaging for final measurements
- Use `style='fine'` for critical parameters
- Balance measurement time vs. precision

#### 4. Configuration Management
- Keep backup copies of working configurations
- Use version control for config files
- Document parameter changes
- Validate config after updates

#### 5. Quality Metrics
Monitor these indicators of measurement quality:
- **R² values**: Should be > 0.8 for good fits
- **Fit errors**: Should be < 10% of parameter values
- **Readout fidelity**: Should be > 80% for reliable measurements
- **Parameter stability**: Values shouldn't drift significantly

#### 6. Automated vs Manual Tuning
- Use `tune_up_qubit()` for routine maintenance
- Use manual scans for troubleshooting
- Combine automated tuning with manual verification
- Customize parameters for specific requirements

### Debugging Tools

#### 1. Fit Quality Checking
```python
# Check fit status
if not experiment.status:
    print("Fit failed!")
    print(f"R² = {experiment.data['r2']}")
    print(f"Fit error = {experiment.data['fit_err']}")

# Plot fit for visual inspection
experiment.display(fit=True, debug=True)
```

#### 2. Parameter Monitoring
```python
# Track parameter evolution
auto_cfg = config.load(cfg_path)
print(f"Current T1: {auto_cfg.device.qubit.T1[qi]} μs")
print(f"Current fidelity: {auto_cfg.device.readout.fidelity[qi]}")
```

#### 3. Measurement Validation
```python
# Repeat measurements for consistency
results = []
for i in range(5):
    t1 = meas.T1Experiment(cfg_dict, qi=qi)
    results.append(t1.data['new_t1'])
print(f"T1 stability: {np.std(results)/np.mean(results)*100:.1f}%")
```

This comprehensive guide should help you understand and effectively use the qubit tuning system. Remember that superconducting qubits are sensitive devices, and parameters can drift over time, so regular re-tuning is essential for maintaining optimal performance.
