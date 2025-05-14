import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict
from gen.qick_experiment import QickExperiment, QickExperiment2DSimple
from gen.qick_program import QickProgram
from qick.asm_v2 import QickSweep1D

import slab_qick_calib.fitting as fitter


class QubitSpecProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        
        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.expt.length
        self.phase = cfg.device.readout.phase[q]
        super()._initialize(cfg, readout="standard")

        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        self.add_loop("freq_loop", cfg.expt.expts)

        if cfg.expt.checkEF:
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 1")

        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        super().measure(cfg)


class QubitSpec(QickExperiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start probe frequency [MHz]
        span: span probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        soft_avgs: number repetitions of experiment sweep
        length: qubit pulse length [us]
        gain: qubit pulse gain [dac units]
        checkEF: flag to check EF transition
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        style="medium",
        min_r2=None,
        max_err=None,
    ):

        # Currently no control of readout time; may want to change for simultaneious readout
        ef = "ef" if "checkEF" in params and params["checkEF"] else ""
        prefix = f"qubit_spectroscopy_{ef}_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Define default parameters
        max_length = 100  # Based on qick error messages, but not investigated
        spec_gain = self.cfg.device.qubit.spec_gain[qi]
        low_gain = self.cfg.device.qubit.low_gain

        if style == 'huge':
            params_def = {"gain": 80*low_gain * spec_gain, "span": 1500, "expts": 1000, "reps": self.reps}
        elif style == "coarse":
            params_def = {"gain": 20*low_gain * spec_gain, "span": 500, "expts": 500, "reps":self.reps}
        elif style == "medium":
            params_def = {"gain": 5*low_gain * spec_gain, "span": 50, "expts": 200, "reps":self.reps}
        elif style == "fine":
            params_def = {"gain": low_gain * spec_gain, "span": 5, "expts": 100, "reps":2*self.reps}
        
        if 'checkEF' in params and params["checkEF"]:
            params_def["gain"] = 3 * params_def["gain"]
            params_def["reps"] = 5 * params_def["reps"]
        params_def2 = {
            "soft_avgs": self.soft_avgs,
            "final_delay": 10,
            "length": 5,
            'readout_length': self.cfg.device.readout.readout_length[qi],
            "pulse_type": "const",
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            'active_reset':False
        }
        params_def = {**params_def2,**params_def}

        # combine params and params_Def, preferring params
        params = {**params_def, **params}

        if params['checkEF']:
            params_def["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
        else:
            params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
        params = {**params_def, **params}

        if params["length"] == "t1":
            if not params['checkEF']:
                params["length"] = 3 * self.cfg.device.qubit.T1[qi]
            else:
                params["length"] = self.cfg.device.qubit.T1[qi] / 4
        if params["length"] > max_length:
            params["length"] = max_length
        if not params['sep_readout']:
            params['readout_length']=params['length']

        self.cfg.expt = params
        # Check for unexpected parameters
        super().check_params(params_def)
    
        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display, progress=progress)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(QubitSpecProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):

        if fit:
            fitterfunc = fitter.fitlor
            fitfunc = fitter.lorfunc
            super().analyze(fitfunc, fitterfunc, use_i=False)
            data["new_freq"] = data["best_fit"][2]
        return self.data

    def display(
        self, fit=True, ax=None, plot_all=True, **kwargs
    ):
        
        fitfunc = fitter.lorfunc
        xlabel = "Qubit Frequency (MHz)"
        
        title = f"Spectroscopy Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain})"
        if self.cfg.expt.checkEF:
            title = "EF " + title

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa
        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]

        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            fitfunc=fitfunc,
            caption_params=caption_params,  # Pass the new structured parameter list
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class QubitSpecPower(QickExperiment2DSimple):
    """
    self.cfg.expt: dict
        - start_f (float): Qubit frequency start [MHz].
        - step_f (float): Frequency step size [MHz].
        - expts_f (int): Number of experiments stepping from start frequency.
        - reps (int): Number of averages per point.
        - soft_avgs (int): Number of start to finish sweeps to average over.
        - length (float): Qubit probe constant pulse length [us].
        - expts_gain (int): Number of gain experiments.
        - max_gain (int): Maximum gain for the sweep.
        - pulse_type (str): Type of pulse, default is 'const'.
        - checkEF (bool): Flag to check EF transition.
        - qubit (int): Qubit index.
        - qubit_chan (int): Qubit channel index.
        - final_delay (float): Relaxation delay [us].
        - log (bool): Flag to indicate if logarithmic gain sweep is used.
        - rng (int): Range for logarithmic gain sweep.
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        params={},
        style="",
        display=True,
        min_r2=None,
        max_err=None,
    ):

        ef = "ef" if "checkEF" in params and params["checkEF"] else ""

        prefix += style + f"qubit_spectroscopy_power_{ef}qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Define default parameters
        if style == "coarse":
            params_def = {"span": 800, "expts": 500}
        elif style == "fine":
            params_def = {"span": 40, "expts": 100}
        else:
            params_def = {"span": 120, "expts": 200}

        params_def2 = {
            "reps": 2*self.reps,
            "rng": 50,
            "max_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 10,
            "log": True,
        }
        params_def = {**params_def, **params_def2}
        params = {**params_def, **params}
        exp_name = QubitSpec 
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params)
        params = {**self.expt.cfg.expt, **params}


        self.cfg.expt = params

        # Check for unexpected parameters
        super().check_params(params_def)

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        if "log" in self.cfg.expt and self.cfg.expt.log == True:
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt["expts_gain"] - 1))

            max_gain = self.cfg.expt["max_gain"]
            gainpts = max_gain * rat ** (np.arange(self.cfg.expt["expts_gain"]))
        else:
            gainpts = self.cfg.expt["start_gain"] + self.cfg.expt[
                "step_gain"
            ] * np.arange(self.cfg.expt["expts_gain"])

        ysweep = [{"pts": gainpts, "var": "gain"}]

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(ysweep, progress=progress)

        return self.data

    def analyze(self, fit=True, **kwargs):
        
        if fit:
            fitterfunc = fitter.fitlor
            super().analyze(fitterfunc=fitterfunc)

        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):

        title = f"Spectroscopy Power Sweep Q{self.cfg.expt.qubit[0]}"
        if self.cfg.expt.checkEF:
            title = f"EF " + title

        xlabel = "Qubit Frequency (MHz)"
        ylabel = "Qubit Gain (DAC level)"

        super().display(
            data=data,
            ax=ax,
            plot_amps=plot_amps,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fit=fit,
            **kwargs,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
