import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
from qick_experiment import QickExperiment, QickExperiment2D
from qick_program import QickProgram
import fitting as fitter
from qick.asm_v2 import QickSweep1D

# ====================================================== #


class AmplitudeRabiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        super()._initialize(cfg, readout="standard")

        if cfg.expt.checkEF:
            sigma_inc = cfg.device.qubit.pulses.pi_ef.sigma_inc[q]
            qubit_freq = cfg.device.qubit.f_ef[q]
        else:
            sigma_inc = cfg.device.qubit.pulses.pi_ge.sigma_inc[q]
            qubit_freq = cfg.device.qubit.f_ge[q]

        pulse = {
            "sigma": cfg.expt.sigma,
            "sigma_inc": sigma_inc,
            "freq": qubit_freq,
            "gain": cfg.expt.gain,
            "phase": 0,
            "type": cfg.expt.pulse_type,
        }
        super().make_pulse(pulse, "qubit_pulse")

        self.add_loop("amp_loop", cfg.expt.expts)
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            pi_pulse = super().make_pi_pulse(q, "pi_ge", cfg.device.qubit.f_ge)
            super().make_pulse(pi_pulse, "pi_pulse_ge")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="qubit_pulse_ge", t=0)

        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="qubit_pulse_ge", t=0)
        self.delay_auto(t=0.01, tag="waiting 2")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )


# ====================================================== #
class AmplitudeRabiExperiment(QickExperiment):
    """
    - 'expts': Number of experiments to run (default: 60)
    - 'reps': Number of repetitions for each experiment (default: self.reps)
    - 'soft_avgs': Number of soft_avgs for each experiment (default: self.soft_avgs)
    - 'gain': Max gain value for the pulse (default: gain)
    - 'sigma': Standard deviation of the Gaussian pulse (default: sigma)
    - 'checkEF': Boolean flag to check EF interaction (default: False)
    - 'checkCC': Boolean flag to check CC interaction (default: False)
    - 'pulse_ge': Boolean flag to indicate if pulse is for ground to excited state transition (default: True)
    - 'start': Starting point for the experiment (default: 0)
    - 'step': Step size for the gain (calculated as int(params['gain']/params['expts']))
    - 'qubit': List of qubits involved in the experiment (default: [qi])
    - 'pulse_type': Type of pulse used in the experiment (default: 'gauss')
    - 'num_pulses': Number of pulses used in the experiment (default: 1)
    - 'qubit_chan': Channel for the qubit readout (default: self.cfg.hw.soc.adcs.readout.ch[qi])
    Additional keys may be added based on the specific requirements of the experiment.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        prefix=None,
        progress=None,
        go=True,
        params={},
        style="",
        checkEF=False,
        min_r2=None,
        max_err=None,
    ):

        if checkEF:
            if params["pulse_ge"]:
                prefix = f"amp_rabi_ef_qubit{qi}"
            else:
                prefix = f"amp_rabi_ef_no_ge_qubit{qi}"
        else:
            prefix = f"amp_rabi_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "checkEF": checkEF,
            "pulse_ge": True,
            "start": 0,
            "qubit": [qi],
            "pulse_type": "gauss",
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}

        if checkEF:
            sigma = self.cfg.device.qubit.pulses.pi_ef.sigma[qi]
            gain = self.cfg.device.qubit.pulses.pi_ef.gain[qi] * 4
        else:
            sigma = self.cfg.device.qubit.pulses.pi_ge.sigma[qi]
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[qi] * 4
        params_def["max_gain"] = gain
        params_def["sigma"] = sigma
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        elif style == "temp":
            params_def["reps"] = 40 * params_def["reps"]
            params_def["soft_avgs"] = 40 * params_def["soft_avgs"]
            params_def["pulse_ge"] = False

        self.cfg.expt = {**params_def, **params}

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit
        self.param = {"label": "qubit_pulse", "param": "gain", "param_type": "pulse"}

        self.cfg.expt.gain = QickSweep1D(
            "amp_loop", self.cfg.expt.start, self.cfg.expt.max_gain
        )
        super().acquire(AmplitudeRabiProgram, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )

        # Get pi length from fit
        ydata_lab = ["amps", "avgi", "avgq"]
        for ydata in ydata_lab:
            pi_length = fitter.fix_phase(data["fit_" + ydata])
            data["pi_gain_" + ydata] = pi_length
        data["pi_gain"] = fitter.fix_phase(data["best_fit"])
        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=False,
        **kwargs,
    ):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]

        title = f"Amplitude Rabi Q{q} (Pulse Length {self.cfg.expt.sigma}"
        xlabel = "Gain / Max Gain"
        fitfunc = fitter.sinfunc
        captionStr = ["$\pi$ gain: {val:.0f}"]
        var = ["pi_gain"]
        if self.cfg.expt.checkEF:
            title = title + ", EF)"
        else:
            title = title + ")"

        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            captionStr=captionStr,
            var=var,
        )

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)


# ====================================================== #


class AmplitudeRabiChevronExperiment(QickExperiment2D):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        soft_avgs: number repetitions of experiment sweep
        sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=None,
        qi=0,
        go=True,
        params={},
        checkEF=False,
        style="",
    ):

        if checkEF:
            prefix = f"amp_rabi_chevron_ef_qubit{qi}"
        else:
            prefix = f"amp_rabi_chevron_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "expts": 60,
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "span_f": 20,
            "expts_f": 40,
            "checkCC": False,
            "checkZZ": False,
            "pulse_type": "gauss",
            "start_gain": 0,
            "pulse_ge": True,
            "qubit": [qi],
            "num_pulses": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        params = {**params_def, **params}

        if checkEF:
            sigma = 2 * self.cfg.device.qubit.pulses.pi_ef.sigma[qi]
            gain = self.cfg.device.qubit.pulses.pi_ef.gain[qi] * 4
            start_f = self.cfg.device.qubit.f_ef[qi] - params["span_f"] / 2
        else:
            sigma = self.cfg.device.qubit.pulses.pi_ge.sigma[qi]
            gain = self.cfg.device.qubit.pulses.pi_ge.gain[qi] * 4
            start_f = self.cfg.device.qubit.f_ge[qi] - params["span_f"] / 2
        gain = int(np.min([gain, self.cfg.device.qubit.max_gain]))
        params_def = {"sigma": sigma, "start_f": start_f, "gain": gain}
        params = {**params_def, **params}
        params["step_f"] = params["span_f"] / (params["expts_f"] - 1)
        params["step_gain"] = int(params["gain"] / params["expts"])
        params["checkEF"] = checkEF

        self.cfg.expt = params

        if go:
            super().run()

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        self.update_config()

        qTest = self.cfg.expt.qubit[0]

        self.cfg.expt.start = self.cfg.expt.start_gain
        self.cfg.expt.step = self.cfg.expt.step_gain
        self.cfg.expt.expts = self.cfg.expt.expts_gain
        if "sigma" not in self.cfg.expt:
            if self.cfg.expt.checkEF:
                self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ef.sigma[qTest]
            else:
                self.cfg.expt.sigma = self.cfg.device.qubit.pulses.pi_ge.sigma[qTest]

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(
            self.cfg.expt["expts_f"]
        )
        ysweep = {"pts": freqpts, "var": "f_pi_test"}
        super().acquire(AmplitudeRabiProgram, ysweep, progress=progress)
        self.cfg.expt["freqpts"] = freqpts

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        title = f"Amplitude Rabi Chevron Q{self.cfg.expt.qubit[0]} (Pulse Length {self.cfg.expt.sigma} $\mu$s)"
        xlabel = "Gain [DAC units]"
        ylabel = "Frequency [MHz]"

        super().display(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            data=data,
            fit=fit,
            plot_both=plot_both,
            **kwargs,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
