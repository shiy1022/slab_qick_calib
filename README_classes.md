# QICK Program and Experiment Classes

This document provides an overview of the base classes for creating quantum experiments using the QICK framework. These classes are designed to be extended for specific experiment implementations.

## QickProgram

The `QickProgram` class is the base for single-qubit experiments. It extends `AveragerProgramV2` from the QICK library to provide a higher-level interface for pulse generation, measurement, and data collection.

### Methods

- `__init__(soccfg, final_delay=50, cfg={})`: Initializes the program with hardware and experiment configurations.
- `_initialize(cfg, readout="standard")`: Sets up ADC and DAC channels, and configures readout and qubit control pulses.
- `_body(cfg)`: Defines the main experiment sequence. This method should be overridden in subclasses.
- `measure(cfg)`: Implements the standard measurement sequence, including readout, LO pulse application, and optional active reset.
- `make_pulse(pulse, name)`: Creates a pulse (Gaussian, flat-top, or constant) and adds it to the program.
- `make_pi_pulse(q, freq, name)`: Creates a π pulse for a specified qubit.
- `collect_shots(offset=0, single=True)`: Retrieves raw I/Q data from the ADC.
- `reset(i)`: Performs active qubit reset by measuring the qubit state and applying a conditional π pulse.

## QickProgram2Q

The `QickProgram2Q` class is the base for two-qubit experiments, extending `AveragerProgramV2` to handle multiple qubits.

### Methods

- `__init__(soccfg, final_delay=50, cfg={})`: Initializes the program for two-qubit experiments.
- `_initialize(cfg, readout="standard")`: Sets up hardware channels for all qubits involved in the experiment.
- `_body(cfg)`: Defines the main experiment sequence. This method should be overridden in subclasses.
- `make_pulse(q, pulse, name)`: Creates a pulse for a specified qubit.
- `make_pi_pulse(q, i, freq, name)`: Creates a π pulse for a specified qubit.
- `collect_shots(offset=[0, 0])`: Retrieves raw I/Q data from all ADCs.
- `reset(i)`: Performs active qubit reset for multiple qubits in parallel.

## QickExperiment

The `QickExperiment` class is the base for running single-shot quantum experiments on QICK hardware. It handles experiment configuration, data acquisition, analysis, and visualization.

### Methods

- `__init__(cfg_dict, qi=0, prefix="QickExp", fname=None, progress=None, check_params=True)`: Initializes the experiment with configuration parameters.
- `acquire(prog_name, progress=True, get_hist=True, single=True, compact=False)`: Acquires data by running a specified `QickProgram`.
- `analyze(fitfunc, fitterfunc, data=None, fit=True, use_i=None, get_hist=True, verbose=True, inds=None, **kwargs)`: Analyzes measurement data by fitting it to a theoretical model.
- `display(data=None, ax=None, plot_all=False, title="", xlabel="", fit=True, show_hist=False, rescale=False, fitfunc=None, caption_params=[], debug=False, **kwargs)`: Displays measurement results with optional fit curves.
- `make_hist(prog, single=True)`: Generates a histogram of single-shot measurement results.
- `qubit_run(qi=0, progress=True, analyze=True, display=True, save=True, print=False, min_r2=0.1, max_err=1, disp_kwargs=None, **kwargs)`: A wrapper for `run` that handles qubit-specific configurations.
- `run(progress=True, analyze=True, display=True, save=True, min_r2=0.1, max_err=1, disp_kwargs=None, **kwargs)`: Runs the complete experiment workflow.
- `save_data(data=None, verbose=False)`: Saves experiment data to disk.
- `print()`: Prints the experimental configuration.
- `get_status(max_err=1, min_r2=0.1)`: Determines if the experiment was successful based on fit quality.
- `get_params(prog)`: Gets swept parameter values from the program.
- `check_params(params_def)`: Checks for unexpected parameters in the configuration.
- `configure_reset()`: Configures parameters for active reset.
- `get_freq(fit=True)`: Calculates the correct frequency when using mixers.
- `scale_ge()`: Scales data to represent excited state probability (0 to 1) based on histogram analysis.

## QickExperimentLoop

The `QickExperimentLoop` class extends `QickExperiment` for loop-based parameter sweeps.

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=False, qi=0)`: Initializes the QickExperimentLoop.
- `acquire(prog_name, x_sweep, progress=True, hist=False)`: Acquires data by running the program for each point in a parameter sweep.
- `stow_data(iq_list, data)`: Processes and stores I/Q data from a measurement.
- `make_hist(shots_i)`: Generates a histogram from collected shots.

## QickExperiment2D

The `QickExperiment2D` class extends `QickExperimentLoop` for 2D parameter sweeps, where one parameter is swept by the program and another by the class.

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=None, qi=0)`: Initializes the QickExperiment2D.
- `acquire(prog_name, y_sweep, progress=True)`: Acquires data for a 2D parameter sweep.
- `analyze(fitfunc, fitterfunc, data=None, fit=False, **kwargs)`: Analyzes 2D data by fitting each row to a model function.
- `display(data=None, ax=None, plot_both=False, plot_amps=False, title="", xlabel="", ylabel="", **kwargs)`: Displays 2D results as a heatmap.

## QickExperiment2DSimple

The `QickExperiment2DSimple` class is a simplified version of `QickExperiment2D` for nested experiments where the x-axis parameter is swept by a separate experiment instance.

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=None, qi=0)`: Initializes the QickExperiment2DSimple.
- `acquire(y_sweep, progress=False)`: Acquires data for a 2D parameter sweep using a nested experiment.

## QickExperiment2DSweep

The `QickExperiment2DSweep` class extends `QickExperiment` for 2D parameter sweeps with a different analysis method.

### Methods

- `analyze(fitfunc, fitterfunc, data=None, fit=False, **kwargs)`: Analyzes 2D data by fitting each row to a model function.
- `display(data=None, ax=None, plot_both=False, plot_amps=False, title="", xlabel="", ylabel="", **kwargs)`: Displays 2D results as a heatmap.
