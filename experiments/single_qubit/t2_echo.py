import matplotlib.pyplot as plt
import numpy as np
from qick import *
from datetime import datetime

from exp_handling.datamanagement import AttrDict
from tqdm import tqdm_notebook as tqdm
from qick_experiment import QickExperiment
import fitting as fitter
from qick_program import QickProgram
from qick.asm_v2 import QickSweep1D


class RamseyEchoProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        cfg_qub = cfg.device.qubit.pulses.pi_ge
        q = cfg.expt.qubit[0]
        pulse = {
            "sigma": cfg_qub.sigma[q]/2,
            "sigma_inc": cfg_qub.sigma_inc[q],
            "freq": cfg.device.qubit.f_ge[q],
            "gain": cfg_qub.gain[q],
            "phase": 0,
            "type": cfg_qub.type,
        }
        super().make_pulse(pulse, "pi2_prep")
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")
        # if cfg.expt.type == "cpmg":
        #     pulse["phase"] = 90
        # elif cfg.expt.type == "cp":
        #     pulse["phase"] = 0
        # else:
        #     assert False, "Unsupported echo experiment type"
        #pulse["gain"] = cfg_qub.gain[q]
        #super().make_pulse(pulse, "pi_pulse")
        super().make_pi_pulse(
            cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge"
        )

        self.add_loop("wait_loop", cfg.expt.expts)
    
    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0)

        self.delay_auto(t=cfg.expt.wait_time/(cfg.expt.num_pi+1) + 0.01, tag="wait")
        for i in range(cfg.expt.num_pi):
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=cfg.expt.wait_time/(cfg.expt.num_pi+1) + 0.01, tag="wait2")
        
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait3")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
            ddr4=True,
        )
        if cfg.expt.active_reset:
            self.reset(3)
        


class RamseyEchoExperiment(QickExperiment):
    """
    Ramsey Echo Experiment
    Experimental Config:
    expt = dict(
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        step: total wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        num_pi: number of pi pulses
        cp: True/False
        cpmg: True/False
        reps: number averages per experiment
        soft_avgs: number soft_avgs to repeat experiment sweep
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        params={},
        style="",
        min_r2=None,
        max_err=None,
        display=True,
    ):
        # span=None, expts=100, ramsey_freq=0.1, reps=None, soft_avgs=None,
        if prefix is None:
            prefix = f"echo_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "reps": 2 * self.reps,
            "soft_avgs": 2 * self.soft_avgs,
            "expts": 100,
            "span": 3 * self.cfg.device.qubit.T2e[qi],
            "start": 0.1,
            "ramsey_freq": 'smart',
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "num_pi": 1,
            "type": "cp",            
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        params = {**params_def, **params}
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = np.pi / 2 / self.cfg.device.qubit.T2e[qi]

        self.cfg.expt = params
        
        if self.cfg.expt.active_reset:
            super().configure_reset()
        super().check_params(params_def)

        if go:
            super().run(display=display,progress=progress,min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        # is this still needed?
        # if self.cfg.expt.ramsey_freq > 0:
        #     self.cfg.expt.ramsey_freq_sign = 1
        # else:
        #     self.cfg.expt.ramsey_freq_sign = -1
        # self.cfg.expt.ramsey_freq_abs = abs(self.cfg.expt.ramsey_freq)
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(RamseyEchoProgram, progress=progress)
        self.data['xpts']=(self.cfg.expt.num_pi+1)*self.data['xpts']

        return self.data

    def analyze(self, data=None, fit=True, debug=False, **kwargs):
        if data is None:
            data = self.data
        if fit:
            fitfunc = fitter.decayslopesin
            fitterfunc = fitter.fitdecayslopesin
            super().analyze(fitfunc, fitterfunc, data, **kwargs)
            

            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data["fit_" + ydata], (list, np.ndarray)):
                    data["f_adjust_ramsey_" + ydata] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                            -self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                        ),
                        key=abs,
                    )

            fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                self.data, get_best_data_params=["f_adjust_ramsey"]
            )

            f_pi_test = self.cfg.device.qubit.f_ge
            data["new_freq"] = f_pi_test + t2r_adjust[0]

        return data

    def display(
        self,
        data=None,
        fit=True,
        debug=False,
        plot_all=False,
        ax=None,
        savefig=True,
        show_hist=False,
        **kwargs,
    ):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]

        xlabel = "Wait Time ($\mu$s)"
        title = f"Ramsey Echo Q{qubit} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"
        fitfunc = fitter.decayslopesin
        caption_params = [
            {"index": 3, "format": "$T_2$ Echo : {val:.4} $\pm$ {err:.2g} $\mu$s"},
            {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},        
        ]


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

        # # Plot the decaying exponential
        # x0 = -(p[2]+180)/360/p[1]
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], p[0], x0, p[3]), color='0.2', linestyle='--')
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], -p[0], x0, p[3]), color='0.2', linestyle='--')

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
