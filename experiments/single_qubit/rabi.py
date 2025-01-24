import matplotlib.pyplot as plt
import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict

import matplotlib.pyplot as plt
from gen.qick_experiment import QickExperiment, QickExperiment2DSimple
from gen.qick_program import QickProgram
import fitting as fitter
from qick.asm_v2 import QickSweep1D
from scipy.optimize import curve_fit
# ====================================================== #


class RabiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        super()._initialize(cfg, readout="standard")

        pulse = {
            "sigma": cfg.expt.sigma,
            "length": cfg.expt.length,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": 0,
            "type": cfg.expt.type,
        }
        super().make_pulse(pulse, "qubit_pulse")

        self.add_loop("sweep_loop", cfg.expt.expts)
        #if cfg.expt.checkEF and cfg.expt.pulse_ge:
        super().make_pi_pulse(q,cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.delay_auto(t=0.01, tag="wait")

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")
        

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(3)

    def reset(self, i):
        super().reset(i)


class RabiExperiment(QickExperiment):
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
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
    ):
        
        if 'checkEF' in params and params['checkEF']:
            if 'pulse_ge' in params and not params['pulse_ge']:
                ef = "ef_no_ge"
            else: ef = "ef"
        else: ef= ""

        prefix = f"amp_rabi_{ef}_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "checkEF": False,
            "pulse_ge": True,
            "sweep": "amp",
            'active_reset': self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }        

        params = {**params_def, **params}
        
        # Pulse params
        if params['checkEF']:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]
        params = {**params_def, **params}
        
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {'plot_all': True}
                
        if params["sweep"]=="amp":
            params_def['max_gain'] = params['gain'] * 4
            params_def['start']=0.003
            params_def["max_gain"]=np.min([params_def["max_gain"], self.cfg.device.qubit.max_gain])
        elif params["sweep"]=="length":
            params_def["sigma"] = 4 * params["sigma"]
            params_def["start"] = 2/430 # Change this to get info from soc 
        
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        elif style == "temp":
            params["reps"] = 100 * params["reps"]
            params["soft_avgs"] = 40 * params["soft_avgs"]
            params["pulse_ge"] = False
        
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if go:
            super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit
        # 2d scans will break if you use gain/length to store the max_vals of gain/length when making qicksweep
        if self.cfg.expt.sweep == "amp":
            param_pulse = 'gain'
            self.cfg.expt['gain'] = QickSweep1D(
        "sweep_loop", self.cfg.expt.start, self.cfg.expt['max_gain']
    )
            self.cfg.expt['length'] = self.cfg.expt.sigma * self.cfg.expt.sigma_inc
        elif self.cfg.expt.sweep == "length":
            param_pulse = 'total_length'
            self.cfg.expt['length'] = QickSweep1D(
        "sweep_loop", self.cfg.expt.start, self.cfg.expt['sigma']
    )
            
        self.param = {"label": "qubit_pulse", "param": param_pulse, "param_type": "pulse"}
        
        super().acquire(RabiProgram, progress=progress)

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
            data["pi_length_" + ydata] = pi_length
        data["pi_length"] = fitter.fix_phase(data["best_fit"])
        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        if self.cfg.expt.sweep == "amp":
            title = 'Amplitude'
            param = 'sigma'
            xlabel = "Gain / Max Gain"
        else:
            title = 'Length'
            param = 'gain'
            xlabel = "Pulse Length ($\mu$s)"
        title += f" Rabi Q{q} (Pulse {param} {self.cfg.expt[param]}"

        fitfunc = fitter.sinfunc
        caption_params = [{'index':"pi_length", 
                          'format':"$\pi$ length: {val:.3f}"}]
        
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
            caption_params=caption_params,
            rescale=rescale,
        )

    def save_data(self, data=None):
        super().save_data(data=data)


class RabiChevronExperiment(QickExperiment2DSimple):
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
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=None,
    ):  
        if 'type' in params:
            pre=params['type']
        else:
            pre="amp"
        if 'checkEF' in params and params['checkEF']: ef = "ef"
        else: ef = ""
        prefix = f"{pre}_rabi_chevron_{ef}_qubit{qi}"         

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            'span_f':20,
            'expts_f': 30,
        }
        params = {**params_def, **params}
        if 'checkEF' in params and params["checkEF"]:
            params_def['start_f'] = self.cfg.device.qubit.f_ef[qi] - params["span_f"] / 2
        else:
            params_def['start_f'] = self.cfg.device.qubit.f_ge[qi] - params["span_f"] / 2
       
        self.expt = RabiExperiment(cfg_dict, qi, go=False, params=params, style=style)
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params
        
        if go:
            super().run()
        

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits

        freqpts = np.linspace(self.cfg.expt["start_f"], self.cfg.expt["start_f"] + self.cfg.expt["span_f"], self.cfg.expt["expts_f"])
        
        ysweep = [{"pts": freqpts, "var": "freq"}]
        super().acquire(ysweep, progress=progress)
        

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data=super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )
            qubit_freq= self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            try:
                p, _ = curve_fit(chevron_freq, data["ypts"]-qubit_freq, freq)
                p2, _ = curve_fit(chevron_amp, data["ypts"]-qubit_freq, amp)
                data["chevron_freq"] = p
                data["chevron_amp"] = p2
            except:
                pass

            
        


    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        if self.cfg.expt.checkEF: 
            title = 'EF'
        else: 
            title = ''
        if self.cfg.expt.type == "amp":
            title = 'Amplitude'
            param = 'sigma'
            xlabel = "Gain / Max Gain"
        else:
            title = 'Length'
            param = 'gain'
            xlabel = "Pulse Length ($\mu$s)"
        title += f" Rabi Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]}"

        xlabel = xlabel
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

        if fit: 
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            qubit_freq= self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            ax[0].plot(data["ypts"]-qubit_freq, freq)
            ax[1].plot(data["ypts"]-qubit_freq, amp)
            ax[1].set_xlabel('$\Delta$ Frequency (MHz)')
            ax[0].set_ylabel('Frequency (MHz)')
            ax[1].set_ylabel('Amplitude')



    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname

def chevron_freq(x, w0): 
    return np.sqrt(w0**2 + x**2)

def chevron_amp(x, w0, a):
    return a/(1 + (x/w0)**2)


