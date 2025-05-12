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

        super()._initialize(cfg, readout="standard")

        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        stark_pulse = {
            "chan": self.res_ch,
            "freq": cfg.expt.stark_freq,
            "gain": cfg.expt.stark_gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(stark_pulse, "stark_pulse")
        self.add_loop("stark_loop", cfg.expt.stark_expts)
        self.add_loop("freq_loop", cfg.expt.expts)


    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.pulse(ch=self.res_ch, name="stark_pulse", t=0)

        self.delay_auto(t=0.01, tag="wait")
        
        super().measure(cfg)


class StarkSpec(QickExperiment):

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
        
        prefix = f"stark_spectroscopy_{style}_qubit{qi}"
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
        
        params_def2 = {
            "soft_avgs": self.soft_avgs,
            "final_delay": 10,
            "length": 5,
            "stark_expts":20,
            "df_stark":0,
            "max_stark_gain":1,
            "stark_rng":10,
            "pulse_type": "const",
            "qubit": [qi],
            "active_reset":False,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params_def = {**params_def2,**params_def}

        # combine params and params_Def, preferring params
        params = {**params_def, **params}


        params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
        params = {**params_def, **params}
        params["stark_freq"]=self.cfg.device.readout.frequency[qi] + params["df_stark"]


        if params["length"] == "t1":
            params["length"] = self.cfg.device.qubit.T1[qi] / 4
        if params["length"] > max_length:
            params["length"] = max_length

        self.cfg.expt = params
        # Check for unexpected parameters
        super().check_params(params_def)
    
        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display, progress=progress, analyze=False)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}
        
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        self.cfg.expt.stark_gain = QickSweep1D(
            "stark_loop", self.cfg.expt.max_stark_gain/self.cfg.expt.stark_rng, self.cfg.expt.max_stark_gain
        )
        super().acquire(QubitSpecProgram, progress=progress, get_hist=False)
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
        
        pass
        # fitfunc = fitter.lorfunc
        # xlabel = "Qubit Frequency (MHz)"
        
        # title = f"Spectroscopy Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain})"
        
        # # Define which fit parameters to display in caption
        # # Index 2 is frequency, index 3 is kappa
        # caption_params = [
        #     {"index": 2, "format": "$f$: {val:.6} MHz"},
        #     {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        # ]

        # super().display(
        #     ax=ax,
        #     plot_all=plot_all,
        #     title=title,
        #     xlabel=xlabel,
        #     fit=fit,
        #     show_hist=False,
        #     fitfunc=fitfunc,
        #     caption_params=caption_params,  # Pass the new structured parameter list
        # )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname