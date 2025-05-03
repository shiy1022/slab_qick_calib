import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict
from gen.qick_experiment import QickExperiment
import slab_qick_calib.fitting as fitter
from gen.qick_program import QickProgram
from qick.asm_v2 import QickSweep1D


class T2Program(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        pulse = {
            "sigma": cfg.expt.sigma / 2,
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

        # For AC Stark shift in Ramsey
        if hasattr(cfg.expt, "acStark") and cfg.expt.acStark:
            pulse = {
                "sigma": cfg.expt.wait_time,
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "const",
            }
            super().make_pulse(pulse, "stark_pulse")

        # For EF check in Ramsey or pi pulse for echo
        if cfg.expt.checkEF or cfg.expt.experiment_type == "echo":
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # For EF check in Ramsey
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        # First pi/2 pulse
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)

        # Handle different experiment types
        if hasattr(cfg.expt, "acStark") and cfg.expt.acStark: # This does not work for echo
            # Ramsey with AC Stark
            self.delay_auto(t=0.01, tag="wait st")
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.025, tag="waiting")
        else:
            self.delay_auto(t=cfg.expt.wait_time / (cfg.expt.num_pi + 1), tag="wait")
            for i in range(cfg.expt.num_pi):
                self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
                self.delay_auto(
                    t=cfg.expt.wait_time / (cfg.expt.num_pi + 1) + 0.01, tag=f"wait{i}"
                )

        # Second pi/2 pulse
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait rd")

        # For EF check in Ramsey
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")

        super().measure(cfg)

    def reset(self, i):
        super().reset(i)


class T2Experiment(QickExperiment):
    """
    T2 Experiment - Supports both Ramsey and Echo protocols

    Experimental Config for Ramsey:
    expt = dict(
        experiment_type: "ramsey" or "echo"
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        soft_avgs: number soft_avgs to repeat experiment sweep
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
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
    ):
       
        # Set default parameters based on experiment type
        if "experiment_type" in params and params["experiment_type"] == "echo":
            par = "T2e"
            name = "echo"
        else:
            par = "T2r"
            name = "ramsey"

        # Set appropriate prefix
        if prefix is None:
            ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
            prefix = f"{name}_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)


        params_def = {
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "expts": 100,
            "span": 3 * self.cfg.device.qubit[par][qi],
            "start": 0.01,
            "ramsey_freq": "smart",
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "experiment_type": "ramsey",
            "acStark": False,
            "checkEF": False,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        # Apply style modifications
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 50
        params = {**params_def, **params}
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = 1.5 / self.cfg.device.qubit[par][qi]
            if style == "fast":
                params["ramsey_freq"] = params["ramsey_freq"]
        if params["experiment_type"] == "echo":
            params_def["num_pi"] = 1
        else:
            params_def["num_pi"] = 0
        # Set pulse parameters
        if "checkEF" in params and params["checkEF"]:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]

        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]


        # Merge default and user-provided parameters
        params = {**params_def, **params}
        self.cfg.expt = params

        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}
        if go:
            super().run(
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(T2Program, progress=progress)

        self.data["xpts"] = (self.cfg.expt.num_pi + 1) * self.data["xpts"]

        return self.data

    def analyze(self, data=None, fit=True, fit_twofreq=False, refit=False,verbose=False, **kwargs):
        if data is None:
            data = self.data

        inds = [0,1,2,3,4]

        if fit:
            if fit_twofreq:
                self.fitterfunc = fitter.fittwofreq_decaysin
                self.fitfunc = fitter.twofreq_decaysin
            if refit:
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
            else:
                self.fitfunc = fitter.decayslopesin
                self.fitterfunc = fitter.fitdecayslopesin
            # yscale, freq, phase_deg, decay, y0, slope

            super().analyze(fitfunc=self.fitfunc, fitterfunc=self.fitterfunc, data=data, inds=inds, **kwargs)

            # It tries to fit to a sloped decaying sine, but if that fails remove the slope. 
            if not self.status and not refit:
                #print('Refitting without slope')
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
                super().analyze(fitfunc=self.fitfunc, fitterfunc=self.fitterfunc, data=data,inds=inds, **kwargs)
                 

            inds = np.arange(5)
            data["fit_err"] = np.mean(np.abs(data["fit_err_par"][inds]))

            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data["fit_" + ydata], (list, np.ndarray)):
                    # -self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],

                    data["f_adjust_ramsey_" + ydata] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                            self.cfg.expt.ramsey_freq + data["fit_" + ydata][1],
                        ),
                        key=abs,
                    )

                    if fit_twofreq and self.cfg.expt.experiment_type == "ramsey":
                        data["f_adjust_ramsey_" + ydata + "2"] = sorted(
                            (
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][7],
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][6],
                            ),
                            key=abs,
                        )

            if not self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]:
                fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                    self.data, get_best_data_params=["f_adjust_ramsey"]
                )
            else:  # ramsey with tuned up qubit
                t2r_adjust = data["f_adjust_ramsey_avgi"]

            data["t2r_adjust"] = t2r_adjust

            if self.cfg.expt.checkEF:
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubit[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]

            if self.cfg.expt.experiment_type == "ramsey" and verbose:
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
        savefig=True,
        refit=False,
        show_hist=False,
        **kwargs,
    ):
        if data is None:
            data = self.data
        q = self.cfg.expt.qubit[0]
        if self.cfg.expt.experiment_type == "echo":
            name = "Echo "
        else:
            name = ""
        # Set up display parameters based on experiment type
        xlabel = "Wait Time ($\mu$s)"

        ef = "EF " if self.cfg.expt.checkEF else ""
        title = f"{ef} Ramsey {name}Q{q} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"

        # Set up caption parameters

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
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
