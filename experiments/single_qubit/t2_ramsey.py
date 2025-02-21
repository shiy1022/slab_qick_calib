import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict

from gen.qick_experiment import QickExperiment
from gen.qick_program import QickProgram
import fitting as fitter
from qick.asm_v2 import QickSweep1D

class RamseyProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        
        pulse = {
            "sigma": cfg.expt.sigma/2,
            "sigma_inc": cfg.expt.sigma_inc,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": 0,
            "type": cfg.expt.type,
        }
        super().make_pulse(pulse, "pi2_prep")
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

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

        #if cfg.expt.checkEF:
        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)

        if cfg.expt.acStark:
            self.delay_auto(t=0.01, tag="wait st")
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.025, tag="waiting")
        else:
            self.delay_auto(t=cfg.expt.wait_time, tag="waiting")
        
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait")

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")

        
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(3)
    
    def collect_shots(self, offset=0):
        return super().collect_shots(offset=0)
    
    def reset(self, i):
        super().reset(i)


class RamseyExperiment(QickExperiment):

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        acStark=False,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
    ):
        ef='ef_' if 'checkEF' in params and params['checkEF'] else''
        prefix = f"ramsey_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "expts": 100,
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "start": 0.1,
            "span": 3 * self.cfg.device.qubit.T2r[qi],
            "ramsey_freq": 'smart',
            "acStark": acStark,
            "checkEF": False,
            "checkZZ": False,
            'active_reset': self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        if params['checkEF']:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = np.pi / 2 / self.cfg.device.qubit.T2r[qi]
        
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        params = {**params_def, **params}
        self.cfg.expt = params
        
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {'plot_all': True}
        if go:
            super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False):

        # if self.cfg.expt.ramsey_freq > 0:
        #     self.cfg.expt.ramsey_freq_sign = 1
        # else:
        #     self.cfg.expt.ramsey_freq_sign = -1
        # self.cfg.expt.ramsey_freq_abs = abs(self.cfg.expt.ramsey_freq)

        self.param = {"label": "waiting", "param": "t", "param_type": "time"}
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
            if fit_twofreq:
                fitterfunc = fitter.fittwofreq_decaysin
            else:
                fitterfunc = fitter.fitdecayslopesin
                fitfunc = fitter.decayslopesin

            super().analyze(fitterfunc=fitterfunc, fitfunc=fitfunc)

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
            if self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]:
                t2r_adjust = data["f_adjust_ramsey_avgi"]
            else:
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
        show_hist=True,
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
            fitfunc = fitter.decayslopesin
        title += f" (Freq: {self.cfg.expt.ramsey_freq:.3f} MHz)"

        caption_params = [
            {"index": 3, "format": "$T_2$ Ramsey : {val:.4} $\pm$ {err:.2g} $\mu$s"},
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
