"""
T2 Measurement Module

This module implements T2 (dephasing time) measurements for superconducting qubits.
T2 is a measure of how long a qubit maintains phase coherence in the x-y plane of the Bloch sphere.

The module supports three main measurement protocols:
1. Ramsey: Uses two π/2 pulses separated by a variable delay time
2. Echo: Uses two π/2 pulses with one or more π pulses in between to refocus dephasing
3. CPMG: Uses a sequence of π pulses to dynamically decouple the qubit from the environment

Additional features include:
- AC Stark shift measurements during Ramsey experiments
- EF transition measurements (first excited to second excited state)
- Automatic frequency error detection and correction
"""

import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D

from ... import fitting as fitter
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict


class T2Program(QickProgram):
    """
    Quantum program for T2 measurements (Ramsey, Echo, and CPMG protocols).

    This class defines the pulse sequences for T2 measurements:
    - Ramsey: π/2 - wait - π/2 sequence to measure phase coherence
    - Echo: π/2 - wait - π - wait - π/2 sequence to refocus dephasing
    - CPMG: π/2 - (wait - π)^n - wait - π/2 sequence for dynamical decoupling

    Additional options include AC Stark shift during wait time and EF transition measurements.
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the T2 program.

        Args:
            soccfg: SOC configuration
            final_delay: Delay after measurement before next experiment
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program by setting up the pulse sequence.

        Creates the necessary pulses for T2 measurements:
        - Two π/2 pulses (prep and read)
        - Optional π pulse(s) for Echo and CPMG
        - Optional AC Stark pulse for Ramsey with AC Stark shift

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Create π/2 pulses for Ramsey/Echo sequence
        # First π/2 pulse has phase=0, second has phase based on wait time and Ramsey frequency
        pulse = {
            "sigma": cfg.expt.sigma / 2,  # Half sigma for π/2 pulse
            "sigma_inc": cfg.expt.sigma_inc,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": 0,  # First pulse has zero phase
            "type": cfg.expt.type,
        }

        # Create first π/2 pulse (preparation)
        super().make_pulse(pulse, "pi2_prep")

        # Create second π/2 pulse (readout) with phase that depends on wait time
        # Phase advances at rate of ramsey_freq (MHz) * wait_time (μs) * 360 (deg/cycle)
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

        # Create loop for sweeping wait time
        self.add_loop("wait_loop", cfg.expt.expts)

        # For AC Stark shift in Ramsey experiments
        if hasattr(cfg.expt, "acStark") and cfg.expt.acStark:
            # Create pulse to apply during wait time
            pulse = {
                "sigma": cfg.expt.wait_time,  # Duration matches wait time
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "const",  # Constant amplitude pulse
            }
            super().make_pulse(pulse, "stark_pulse")

        # Create π pulse for Echo or EF check
        if cfg.expt.experiment_type == "cpmg":
            cfg.device.qubit.pulses.pi_ge.phase = 90 * np.ones(
                len(cfg.device.qubit.f_ge)
            )
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        elif (
            cfg.expt.checkEF
            or cfg.expt.experiment_type == "echo"
            or cfg.expt.active_reset
        ):
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        if cfg.expt.checkEF and cfg.expt.experiment_type == "echo":
            # Create π pulse for EF transition check
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

        for i in range(1000):
            self.nop()

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        Implements the actual T2 measurement sequence:
        - Ramsey: π/2 - wait - π/2
        - Echo: π/2 - wait/2 - π - wait/2 - π/2
        - With options for AC Stark and EF measurements

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # For EF transition check in Ramsey: Apply π pulse to excite |g⟩ to |e⟩ first
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")  # Small buffer delay

        # First π/2 pulse (preparation)
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)

        # Handle different experiment types
        if (
            hasattr(cfg.expt, "acStark") and cfg.expt.acStark
        ):  # Ramsey with AC Stark shift
            # Note: AC Stark shift is not compatible with Echo protocol
            self.delay_auto(t=0.01, tag="wait st")  # Small buffer delay
            self.pulse(
                ch=self.qubit_ch, name="stark_pulse", t=0
            )  # Apply AC Stark pulse
            self.delay_auto(t=0.025, tag="waiting")  # Additional wait time
        else:
            # Standard Ramsey or Echo sequence
            # For Echo, divide wait time by (num_pi + 1) to get segments between pulses
            if cfg.expt.num_pi > 0:
                self.delay_auto(t=cfg.expt.wait_time / cfg.expt.num_pi / 2, tag="wait")

                # Apply π pulses for Echo protocol (or multiple-pulse Echo)
                for i in range(cfg.expt.num_pi):
                    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)  # π pulse
                    if i < cfg.expt.num_pi - 1:
                        self.delay_auto(
                            t=cfg.expt.wait_time / cfg.expt.num_pi + 0.01,
                            tag=f"wait{i}",
                        )  # Wait time
                self.delay_auto(
                    t=cfg.expt.wait_time / cfg.expt.num_pi / 2 + 0.01, tag=f"wait{i+1}"
                )
            else:
                self.delay_auto(t=cfg.expt.wait_time, tag="wait")

        # Second π/2 pulse (readout)
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait rd")  # Small buffer delay

        # For EF transition check in Ramsey: Apply π pulse to return to |g⟩ for readout
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")  # Small buffer delay

        # Measure the qubit state
        super().measure(cfg)


class T2Experiment(QickExperiment):
    """
    T2 Experiment - Supports Ramsey, Echo, and CPMG protocols

    Experimental Config for Ramsey:
    expt = dict(
        experiment_type: "ramsey", "echo", or "cpmg"
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        acStark: True/False (Ramsey only)
        checkEF: True/False (Ramsey only)
    )

    Additional config for Echo:
    expt = dict(
        num_pi: number of pi pulses
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        fname=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        print=False,
    ):
        """
        Initialize the T2 experiment.

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index to measure
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            prefix: Filename prefix for saved data
            progress: Whether to show progress bar
            style: Measurement style ('fine' for more averages, 'fast' for fewer points)
            disp_kwargs: Display options
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            display: Whether to display results
        """
        # Determine experiment type and parameter name based on protocol
        if "experiment_type" in params and params["experiment_type"] == "echo":
            par = "T2e"  # Echo uses T2e parameter
            name = "echo"
        else:
            par = "T2r"  # Ramsey uses T2r parameter
            name = "ramsey"

        # Set appropriate filename prefix
        if prefix is None:
            ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
            prefix = f"{name}_{ef}qubit{qi}"

        # Initialize parent class
        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, fname=fname, progress=progress, qi=qi
        )

        # Define default parameters
        params_def = {
            "reps": 2 * self.reps,  # Number of repetitions (inner loop)
            "rounds": self.rounds,  # Number of averages (outer loop)
            "expts": 100,  # Number of wait time points
            "span": 3
            * self.cfg.device.qubit[par][
                qi
            ],  # Total span of wait times (μs), set to ~3*T2
            "start": 0.01,  # Start time for wait sweep (μs)
            "ramsey_freq": "smart",  # Ramsey frequency for phase advancement
            "active_reset": self.cfg.device.readout.active_reset[
                qi
            ],  # Use active qubit reset
            "qubit": [qi],  # Qubit index as a list
            "experiment_type": "ramsey",  # Default to Ramsey protocol
            "acStark": False,  # No AC Stark shift by default
            "checkEF": False,  # No EF transition check by default
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
        }

        # Adjust parameters based on measurement style
        if style == "fine":
            params_def["rounds"] = (
                params_def["rounds"] * 2
            )  # Double averages for fine measurements
        elif style == "fast":
            params_def["expts"] = 50  # Fewer points for fast measurements

        # Merge user parameters with defaults
        params = {**params_def, **params}

        # Set Ramsey frequency intelligently if "smart" is specified
        if params["ramsey_freq"] == "smart":
            # Set Ramsey frequency to 1.5/T2 for optimal oscillation visibility
            params["ramsey_freq"] = 1.5 / self.cfg.device.qubit[par][qi]

        # Set number of π pulses based on experiment type
        if params["experiment_type"] == "echo":
            params_def["num_pi"] = 1  # Standard echo has 1 π pulse
        else:
            params_def["num_pi"] = 0  # Ramsey has 0 π pulses

        # Set pulse parameters based on transition type (g-e or e-f)
        if "checkEF" in params and params["checkEF"]:
            # For e-f transition measurements
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            # For g-e transition measurements (standard)
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]

        # Copy pulse parameters from configuration
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]

        # Final parameter merge and assignment
        params = {**params_def, **params}
        self.cfg.expt = params

        # Check for unexpected parameters
        super().check_params(params_def)

        if go:
            super().qubit_run(
                qi=qi,
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                print=print,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False):
        """
        Acquire T2 measurement data.

        This method:
        1. Sets up the wait time sweep parameters
        2. Runs the T2Program to collect data for each wait time
        3. Adjusts x-axis values to account for echo protocol

        Args:
            progress: Whether to show progress bar

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}

        # Create a 1D sweep for the wait time from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Run the T2Program to acquire data

        super().acquire(T2Program, progress=progress)

        # Adjust x-axis values to account for echo protocol
        # For echo, the effective wait time is longer due to the π pulses
        if self.cfg.expt.num_pi == 0:
            coef = 1
        else:
            coef = self.cfg.expt.num_pi  # For echo, we have num_pi + 1 segments
        self.data["xpts"] = coef * self.data["xpts"]

        return self.data

    def analyze(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        refit=False,
        verbose=False,
        **kwargs,
    ):
        """
        Analyze T2 measurement data by fitting to a decaying sinusoid.

        This method:
        1. Fits the data to a decaying sinusoid (with or without slope)
        2. Calculates frequency errors from the fit
        3. Determines the corrected qubit frequency

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            fit_twofreq: Whether to fit to a two-frequency model
            refit: Whether to refit without slope
            verbose: Whether to print detailed information
            **kwargs: Additional arguments passed to the analyzer

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Define indices for fit parameters
        inds = [0, 1, 2, 3, 4]  # yscale, freq, phase_deg, decay, y0

        if fit:
            # Select appropriate fitting function based on parameters
            if fit_twofreq:
                # Two-frequency model for more complex oscillations
                self.fitterfunc = fitter.fittwofreq_decaysin
                self.fitfunc = fitter.twofreq_decaysin
            elif refit:
                # Simple decaying sine without slope
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
            else:
                # Decaying sine with slope (default)
                self.fitfunc = fitter.decayslopesin
                self.fitterfunc = fitter.fitdecayslopesin
                # Parameters: yscale, freq, phase_deg, decay, y0, slope

            # Perform the fit
            super().analyze(
                fitfunc=self.fitfunc,
                fitterfunc=self.fitterfunc,
                data=data,
                inds=inds,
                **kwargs,
            )

            # If the fit fails, try again without slope
            if not self.status and not refit:
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
                super().analyze(
                    fitfunc=self.fitfunc,
                    fitterfunc=self.fitterfunc,
                    data=data,
                    inds=inds,
                    **kwargs,
                )

            # Calculate average fit error
            inds = np.arange(5)
            data["fit_err"] = np.mean(np.abs(data["fit_err_par"][inds]))

            # Calculate frequency adjustments for each data type (amps, avgi, avgq)
            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data["fit_" + ydata], (list, np.ndarray)):
                    # Calculate possible frequency errors
                    # The fitted frequency can be either ramsey_freq + fit_freq or ramsey_freq - fit_freq
                    data["f_adjust_ramsey_" + ydata] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                            self.cfg.expt.ramsey_freq + data["fit_" + ydata][1],
                        ),
                        key=abs,  # Sort by absolute value to get the smallest error first
                    )

                    # For two-frequency model, calculate additional adjustments
                    if fit_twofreq and self.cfg.expt.experiment_type == "ramsey":
                        data["f_adjust_ramsey_" + ydata + "2"] = sorted(
                            (
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][7],
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][6],
                            ),
                            key=abs,
                        )

            # Get the best frequency adjustment
            if not self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]:
                # For untuned qubits, use a more sophisticated method to find the best fit
                fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                    self.data, get_best_data_params=["f_adjust_ramsey"]
                )
            else:
                # For tuned qubits, use the I quadrature adjustment
                t2r_adjust = data["f_adjust_ramsey_avgi"]

            # Store the frequency adjustment
            data["t2r_adjust"] = t2r_adjust

            # Get the reference frequency based on transition type
            if self.cfg.expt.checkEF:
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubit[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]

            # Print possible frequency errors if verbose
            if self.cfg.expt.experiment_type == "ramsey" and verbose:
                print(
                    f"Possible errors are {t2r_adjust[0]:.3f} and {t2r_adjust[1]:.3f} MHz "
                    f"for Ramsey frequency {self.cfg.expt.ramsey_freq:.3f} MHz"
                )

            # Store the frequency error and corrected frequency
            data["f_err"] = t2r_adjust[0]  # Use the smallest error
            data["new_freq"] = (
                f_pi_test + t2r_adjust[0]
            )  # Calculate corrected frequency

        return data

    def display(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        debug=False,
        plot_all=False,
        ax=None,
        savefig=True,
        refit=False,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        """
        Display T2 measurement results.

        Creates a plot showing the qubit state vs wait time and the exponentially decaying sinusoidal fit.

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show the fit curve
            fit_twofreq: Whether to use two-frequency model for display
            debug: Whether to show debug information
            plot_all: Whether to make plots for I/Q/Amps or just I
            ax: Matplotlib axis to plot on (creates one if None)
            savefig: Whether to save the figure to disk
            refit: Whether to use refit data for display
            show_hist: Whether to show histogram of the data
            **kwargs: Additional arguments passed to the display function
        """
        if data is None:
            data = self.data

        # Get qubit index for plot title
        q = self.cfg.expt.qubit[0]

        # Set experiment name based on type
        name = "Echo " if self.cfg.expt.experiment_type == "echo" else ""
        if self.cfg.expt.num_pi > 1:
            name += f"{self.cfg.expt.num_pi} π pulses "

        # Set x-axis label
        xlabel = "Wait Time ($\mu$s)"

        # Add EF prefix if checking EF transition
        ef = "EF " if self.cfg.expt.checkEF else ""

        # Create plot title
        title = f"{ef} Ramsey {name}Q{q} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"

        # Set up caption parameters to display T2 and frequency values
        if self.cfg.expt.experiment_type == "echo":
            caption_params = [
                {"index": 3, "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s"},
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]
        else:  # ramsey
            caption_params = [
                {
                    "index": 3,
                    "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s",
                },
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]

        # Call parent class display method
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            debug=debug,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            savefig=savefig,
            rescale=rescale,
        )
