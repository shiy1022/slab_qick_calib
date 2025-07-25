# Data Structures Reference: obj.data Contents

This document describes what data is stored in the `obj.data` dictionary for each experiment type, where `obj` is the name of each scan/experiment instance.

## Overview

All experiments in this codebase inherit from the `QickExperiment` base class, which provides standardized `acquire()` and `analyze()` methods. These methods create a common `data` dictionary structure that is then extended by specific experiment types with their own specialized fields.

## Common Data Fields (All Experiments)

Every experiment's `obj.data` dictionary contains these base fields from the `QickExperiment.acquire()` method:

### Raw Measurement Data
- **`xpts`** (array): Swept parameter values (e.g., time, frequency, amplitude)
- **`avgi`** (array): I quadrature measurements (real part)
- **`avgq`** (array): Q quadrature measurements (imaginary part)  
- **`amps`** (array): Amplitude data calculated as `|I + jQ|`
- **`phases`** (array): Phase data calculated as `angle(I + jQ)`

### Histogram Data (if `get_hist=True`)
- **`bin_centers`** (array): Histogram bin center values
- **`hist`** (array): Histogram counts for single-shot measurements

### Metadata
- **`start_time`** (bytes): Timestamp when experiment started
- **`attrs`** (dict): File attributes including the complete experiment configuration

### Configuration Data (in `attrs`)
When data is loaded from file using `load_data()`, the `attrs` field contains the complete experiment configuration:
- **`attrs['config']`** (JSON string): Complete configuration including:
  - **Device configuration**: Qubit parameters, readout settings, hardware configuration
  - **Experiment parameters**: All parameters used for the specific experiment (stored in `cfg.expt`)
  - **Hardware settings**: SOC configuration, channel mappings, mixer frequencies
  - **Calibration data**: Current qubit frequencies, pulse parameters, T1/T2 values

To access the configuration:
```python
import json
config = json.loads(data['attrs']['config'])
# Access device parameters
qubit_freq = config['device']['qubit']['f_ge'][0]  # Qubit 0 frequency
t1_time = config['device']['qubit']['T1'][0]       # Qubit 0 T1 time
# Access experiment parameters  
exp_params = config['expt']                        # Experiment-specific parameters
```

### Fit Results (if `analyze()` called with `fit=True`)
- **`fit_amps`** (array): Fit parameters for amplitude data
- **`fit_avgi`** (array): Fit parameters for I quadrature data
- **`fit_avgq`** (array): Fit parameters for Q quadrature data
- **`fit_err_amps`** (matrix): Covariance matrix for amplitude fit
- **`fit_err_avgi`** (matrix): Covariance matrix for I quadrature fit
- **`fit_err_avgq`** (matrix): Covariance matrix for Q quadrature fit
- **`fit_init_amps`** (array): Initial guess parameters for amplitude fit
- **`fit_init_avgi`** (array): Initial guess parameters for I quadrature fit
- **`fit_init_avgq`** (array): Initial guess parameters for Q quadrature fit
- **`best_fit`** (array): Best fit parameters (automatically selected)
- **`i_best`** (string): Which data type gave the best fit ('amps', 'avgi', or 'avgq')
- **`r2`** (float): R-squared goodness of fit metric
- **`fit_err`** (float): Mean relative parameter error
- **`fit_err_par`** (array): Relative error for each fit parameter

### Scaled Data (if histogram analysis performed)
- **`scale_data`** (array): Data scaled from 0 (ground state) to 1 (excited state)
- **`hist_fit`** (array): Fit parameters for histogram (two-Gaussian model)

---

## T1 Experiments

### T1Experiment
Measures energy relaxation time by applying a π pulse and measuring decay.

**Additional Data Fields:**
- **`new_t1`** (float): T1 time extracted from best combined I/Q fit (μs)
- **`new_t1_i`** (float): T1 time extracted from I quadrature fit (μs)

**Fit Function:** Exponential decay: `f(t) = y_offset + amp * exp(-t/T1)`
**Fit Parameters:** `[y_offset, amplitude, T1_time]`

### T1_2D  
2D version that tracks T1 over time for stability analysis.

**Additional Data Fields:**
- **`ypts`** (array): Time points for the second dimension (hours)
- **`time`** (array): Absolute timestamps for each measurement
- All fit arrays become 2D: `fit_avgi[i][j]` where i is time index, j is parameter index

---

## T2 Experiments

### T2Experiment
Measures dephasing time using Ramsey, Echo, or CPMG protocols.

**Additional Data Fields:**
- **`f_adjust_ramsey_amps`** (array): Possible frequency corrections from amplitude fit [MHz]
- **`f_adjust_ramsey_avgi`** (array): Possible frequency corrections from I fit [MHz] 
- **`f_adjust_ramsey_avgq`** (array): Possible frequency corrections from Q fit [MHz]
- **`t2r_adjust`** (array): Best frequency adjustment values [MHz]
- **`f_err`** (float): Frequency error (smallest correction needed) [MHz]
- **`new_freq`** (float): Corrected qubit frequency [MHz]

**For Two-Frequency Fits:**
- **`f_adjust_ramsey_amps2`** (array): Second frequency corrections from amplitude fit
- **`f_adjust_ramsey_avgi2`** (array): Second frequency corrections from I fit
- **`f_adjust_ramsey_avgq2`** (array): Second frequency corrections from Q fit

**Fit Function:** Decaying sinusoid: `f(t) = y_offset + amp * exp(-t/T2) * sin(2πft + φ) + slope*t`
**Fit Parameters:** `[amplitude, frequency, phase_deg, T2_time, y_offset, slope]`

---

## Rabi Experiments

### RabiExperiment
Measures Rabi oscillations by sweeping pulse amplitude or length.

**Additional Data Fields:**
- **`pi_length_amps`** (float): π-pulse length from amplitude fit
- **`pi_length_avgi`** (float): π-pulse length from I quadrature fit  
- **`pi_length_avgq`** (float): π-pulse length from Q quadrature fit
- **`pi_length_scale_data`** (float): π-pulse length from scaled data fit
- **`pi_length`** (float): Best π-pulse length (from best fit)

**Fit Function:** Sinusoid: `f(x) = y_offset + amp * sin(2πfx + φ)`
**Fit Parameters:** `[amplitude, frequency, phase_deg, y_offset]`

### RabiChevronExperiment
2D Rabi experiment sweeping both frequency and amplitude/length.

**Additional Data Fields:**
- **`ypts`** (array): Frequency points for the second dimension [MHz]
- **`chevron_freqs`** (array): Fitted Rabi frequencies for each detuning [MHz]
- **`chevron_amps`** (array): Fitted Rabi amplitudes for each detuning
- **`best_freq`** (float): Frequency with maximum Rabi amplitude [MHz]
- **`chevron_freq`** (array): Fit parameters for frequency vs detuning
- **`chevron_amp`** (array): Fit parameters for amplitude vs detuning

### Rabi2D
2D Rabi experiment sweeping both length and gain.

**Additional Data Fields:**
- **`ypts`** (array): Gain points for the second dimension
- **`gain_pts`** (array): All gain values used in the sweep
- Similar chevron analysis fields as RabiChevronExperiment

---

## Spectroscopy Experiments

### ResSpec (Resonator Spectroscopy)
Finds resonator frequency by sweeping readout frequency.

**Additional Data Fields:**
- **`freq`** (array): Actual frequencies including mixer offsets [MHz]
- **`freq_offset`** (float): Frequency offset from mixers [MHz]
- **`phase_fix`** (array): Phase data with linear slope removed
- **`kappa`** (float): Resonator linewidth [MHz]
- **`freq_fit`** (array): Fit parameters in absolute frequency
- **`freq_init`** (array): Initial fit guess in absolute frequency  
- **`freq_min`** (float): Frequency with minimum transmission [MHz]

**For Peak Finding (coarse scans):**
- **`coarse_peaks_index`** (array): Indices of identified peaks
- **`coarse_peaks`** (array): Frequencies of identified peaks [MHz]
- **`coarse_props`** (dict): Peak properties (width, prominence, etc.)

**Fit Function:** Hanger model: `S21(f) = scale * (1 + slope*f) * (1 - Q/Qe * e^(jφ) / (1 + 2jQ*(f-f0)/f0))`
**Fit Parameters:** `[f0, Qi, Qe, phi, scale, slope]`

### ResSpecPower
2D resonator spectroscopy sweeping frequency and power.

**Additional Data Fields:**
- **`ypts`** (array): Power/gain points for the second dimension
- **`gain_pts`** (array): All gain values used in the sweep
- **`fit`** (list): Fit results for [high_power, low_power] data
- **`fit_gains`** (array): Gain values used for fitting [high, low]
- **`lamb_shift`** (float): Lamb shift (frequency difference) [MHz]
- **`freq`** (array): Resonator frequencies [f_high, f_low] [MHz]
- **`kappa`** (array): Resonator linewidths [kappa_high, kappa_low] [MHz]

### ResSpec2D
2D resonator spectroscopy for repeated measurements over time.

**Additional Data Fields:**
- **`ypts`** (array): Time points in hours
- **`avgi_full`** (array): Full 2D I quadrature data before averaging
- **`avgq_full`** (array): Full 2D Q quadrature data before averaging  
- **`amps_full`** (array): Full 2D amplitude data before averaging
- **`phases_full`** (array): Full 2D phase data before averaging
- **`phase_raw`** (array): Raw phase data before processing
- **`phase_fix`** (array): Phase data with linear slope removed

---

## 2D Experiment Data Structure

All 2D experiments (T1_2D, RabiChevronExperiment, ResSpecPower, etc.) share this pattern:

### Dimensions
- **`xpts`** (array): First dimension sweep values (e.g., time, frequency)
- **`ypts`** (array): Second dimension sweep values (e.g., power, time)

### 2D Data Arrays
All measurement arrays become 2D with shape `[len(ypts), len(xpts)]`:
- **`avgi`** (2D array): I quadrature data
- **`avgq`** (2D array): Q quadrature data  
- **`amps`** (2D array): Amplitude data
- **`phases`** (2D array): Phase data

### 2D Fit Results
When fitting is performed, fit parameters become lists of arrays:
- **`fit_avgi`** (list): List of fit parameters for each y-point
- **`fit_err_avgi`** (list): List of covariance matrices for each y-point

---

## Data Access Examples

```python
# Basic data access
t1_data = t1_obj.data
times = t1_data['xpts']  # Wait times in μs
signal = t1_data['avgi']  # I quadrature measurements
t1_time = t1_data['new_t1']  # Fitted T1 time

# Fit parameter access
fit_params = t1_data['fit_avgi']  # [y_offset, amplitude, T1_time]
t1_from_fit = fit_params[2]  # T1 time is the 3rd parameter
fit_error = t1_data['fit_err_avgi']  # Covariance matrix

# 2D data access
rabi_2d_data = rabi_chevron_obj.data
frequencies = rabi_2d_data['ypts']  # Frequency sweep
amplitudes = rabi_2d_data['xpts']   # Amplitude sweep  
signal_2d = rabi_2d_data['avgi']    # 2D array [freq_idx, amp_idx]

# Accessing fit for specific frequency point
freq_idx = 5
fit_for_freq = rabi_2d_data['fit_avgi'][freq_idx]  # Fit params for this frequency
```

## Notes

1. **Array Indexing**: Most fits exclude the first and last data points (`[1:-1]`) to avoid edge effects.

2. **Fit Parameter Order**: The order of fit parameters depends on the specific fitting function used. Check the experiment's `fitfunc` and `fitterfunc` for details.

3. **Units**: 
   - Time: microseconds (μs)
   - Frequency: megahertz (MHz) 
   - Amplitude: ADC units
   - Phase: radians

4. **Best Fit Selection**: The `best_fit` field contains parameters from whichever data type (amps, avgi, avgq) gave the best R² value.

5. **Error Handling**: If fitting fails, fit-related fields may not be present in the data dictionary.

6. **Histogram Data**: Only present if `get_hist=True` was used during acquisition.
