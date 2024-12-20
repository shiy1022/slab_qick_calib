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


class RamseyProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        cfg_qub = cfg.device.qubit.pulses.pi_ge
        q = cfg.expt.qubit[0]
        pulse = {
            "sigma": cfg_qub.sigma[q],
            "sigma_inc": cfg_qub.sigma_inc[q],
            "freq": cfg.device.qubit.f_ge[q],
            "gain": cfg_qub.gain[q] / 2,
            "phase": 0,
            "type": cfg_qub.type,
        }
        pi2_pulse1 = super().make_pulse(pulse, "pi2_prep")
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.ramsey_freq
        pi2_pulse2 = super().make_pulse(pulse, "pi2_read")

        self.add_loop("wait_loop", cfg.expt.expts)

        if cfg.expt.acStark:
            pulse = {
                "sigma": cfg.expt.wait_time,
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "const",
            }
            super().make_pulse(pulse, "stark_pulse")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0)
        if cfg.expt.acStark:
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.01, tag="waiting")
        else:
            self.delay_auto(t=cfg.expt.wait_time + 0.01, tag="wait")
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=cfg.device.readout.trig_offset[cfg.expt.qubit[0]],
        )


class RamseyExperiment(QickExperiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        soft_avgs: number soft_avgs to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
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
        check_ef=False,
        style="",
        min_r2=None,
        max_err=None,
    ):
        # span=None, expts=100, ramsey_freq=0.1, reps=None, soft_avgs=None,
        if check_ef:
            prefix = f"ramsey_ef_qubit{qi}"
        else:
            prefix = f"ramsey_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "expts": 100,
            "reps": 2 * self.reps,
            "soft_avgs": 2 * self.soft_avgs,
            "start": 0.1,
            "span": 3 * self.cfg.device.qubit.T2e[qi],
            "ramsey_freq": 0.1,
            "acStark": False,
            "checkEF": check_ef,
            "checkZZ": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = np.pi / 2 / self.cfg.device.qubit.T2e[qi]
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.update_config()

        if self.cfg.expt.ramsey_freq > 0:
            self.cfg.expt.ramsey_freq_sign = 1
        else:
            self.cfg.expt.ramsey_freq_sign = -1
        self.cfg.expt.ramsey_freq_abs = abs(self.cfg.expt.ramsey_freq)

        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(RamseyProgram, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, fit_twofreq=False, debug=False, **kwargs):
        if data is None:
            data = self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # fitparams=[yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01] # two fit freqs
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None
            if fit_twofreq:
                fitterfunc = fitter.fittwofreq_decaysin
            else:
                fitterfunc = fitter.fitdecaysin
            fitfunc = fitter.decaysin

            super().fit_data(fitterfunc=fitterfunc, fitfunc=fitfunc)

            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                data["f_adjust_ramsey_" + ydata] = sorted(
                    (
                        self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                        self.cfg.expt.ramsey_freq + data["fit_" + ydata][1],
                    ),
                    key=abs,
                )
                if fit_twofreq:
                    data["f_adjust_ramsey_" + ydata + "2"] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][7],
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][6],
                        ),
                        key=abs,
                    )

            fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                self.data, get_best_data_params=["f_adjust_ramsey"]
            )
            data["t2r_adjust"] = t2r_adjust

            if self.cfg.expt.checkEF:
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubit[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]

            print(
                f"Possible errors are {t2r_adjust[0]:.3f} and {t2r_adjust[1]:.3f} for Ramsey frequency {self.cfg.expt.ramsey_freq:.3f} MHz"
            )
            data["f_err"] = t2r_adjust[0]
            data["new_freq"] = f_pi_test + t2r_adjust[0]

        return data

    def display(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        debug=False,
        plot_all=False,
        ax=None,
        show_hist=False,
        **kwargs,
    ):
        if data is None:
            data = self.data
        q = self.cfg.expt.qubit[0]

        self.qubits = self.cfg.expt.qubit
        self.checkEF = self.cfg.expt.checkEF

        title = ("EF" if self.checkEF else "") + f"Ramsey Q{q}"
        xlabel = "Wait Time ($\mu$s)"
        if fit_twofreq:
            fitfunc = fitter.twofreq_decaysin
        else:
            fitfunc = fitter.decaysin
        title += f" (Freq: {self.cfg.expt.ramsey_freq:.3f} MHz)"
        captionStr = [
            "$T_2$ Ramsey : {val:.4} $\pm$ {err:.2g} $\mu$s",
            "Freq. : {val:.3} $\pm$ {err:.1} MHz",
        ]
        var = [3, 1]
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

        # # Plot the decaying exponential
        # x0 = -(p[2]+180)/360/p[1]
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], p[0], x0, p[3]), color='0.2', linestyle='--')
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], -p[0], x0, p[3]), color='0.2', linestyle='--')

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
