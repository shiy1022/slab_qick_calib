import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import fitting as fitter
from qick_experiment import QickExperiment, QickExperiment2D
from qick_program import QickProgram
from qick.asm_v2 import QickSweep1D


class T1MultiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        for i in range(len(self.cfg.expt.qubit)):
            super().make_pi_pulse(
                cfg.expt.qubit[i], cfg.device.qubit.f_ge, f"pi_ge_q{i}"
            )
        
        self.add_loop("wait_loop", cfg.expt.expts)

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(t=cfg.expt["wait_time"], tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )

    def collect_shots(self, offset=0):
        return super().collect_shots(offset=0)

class T1MultiExperiment(QickExperiment):
    """
    self.cfg.expt: dict
        A dictionary containing the configuration parameters for the T1 experiment. The keys and their descriptions are as follows:
        - span (float): The total span of the wait time sweep in microseconds.
        - expts (int): The number of experiments to be performed.
        - reps (int): The number of repetitions for each experiment (inner loop)
        - soft_avgs (int): The number of soft_avgs for the experiment (outer loop)
        - qubit (int): The index of the qubit being used in the experiment.
        - qubit_chan (int): The channel of the qubit being read out.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "expts": 60,
            "start": 0,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "acStark": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(T1MultiProgram, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        data["new_t1"] = data["best_fit"][2]
        data["new_t1_i"] = data["fit_avgi"][2]
        return data

    def display(
        self, data=None, fit=True, plot_all=False, ax=None, show_hist=True, **kwargs
    ):
        qubit = self.cfg.expt.qubit[0]
        title = f"$T_1$ Q{qubit}"
        xlabel = "Wait Time ($\mu$s)"


        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},           
        ]
        fitfunc = fitter.expfunc

        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            caption_params=caption_params,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
