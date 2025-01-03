import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from qick_experiment import QickExperiment2DSimple, QickExperiment
import experiments as meas
from qick.asm_v2 import QickSweep1D
from scipy.optimize import curve_fit

import fitting as fitter

class RamseyStarkExperiment(QickExperiment):
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
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        acStark=True,
        style="",
        min_r2=None,
        max_err=None,
    ):
        if prefix is None:
            prefix = f"ramsey_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "expts": 200,
            "reps": 2 * self.reps,
            "soft_avgs": 2 * self.soft_avgs,
            "start": 0.1,
            "ramsey_freq": 'smart',
            "stark_gain": 0.5,
            "step": 0.0023251488095238095,
            "df": 70,
            "acStark": acStark,
            "checkEF": False,
            "checkZZ": False,
            'active_reset': False,
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
            params_def[key] = self.cfg.device.qubit.pulses.pi_ef[key][qi]
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = np.pi / 2 / self.cfg.device.qubit.T2r[qi]
        
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        self.cfg.expt = params
        
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        self.param = {"label": "waiting", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.step*self.cfg.expt.expts
        )

        data = super().acquire(meas.RamseyProgram, progress=progress)
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        fitterfunc = fitter.fitdecaysin
        fitfunc = fitter.decaysin
        if fit:
            super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data)

        return self.data

    def display(
        self, data=None, fit=True, debug=False, plot_all=False, ax=None, show_hist=False, **kwargs
    ):
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]
        title = (
            f"$T_2$ Ramsey Stark Q{qubit} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        )
        xlabel = "Wait Time ($\mu$s)"
        
        fitfunc = fitter.decaysin

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

        return data

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class RamseyStarkFreqExperiment(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
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
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        soccfg=None,
        path="",
        prefix="RamseyStarkFreq",
        config_file=None,
        progress=None,
        im=None,
    ):
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
            im=im,
        )

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})

        if self.cfg.expt.checkZZ:
            assert len(self.cfg.expt.qubits) == 2
            qZZ, qTest = self.cfg.expt.qubits
            assert qZZ != 1
            assert qTest == 1
        else:
            qTest = self.cfg.expt.qubits[0]

        freqpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(
            self.cfg.expt["expts_f"]
        )
        data = {
            "xpts": [],
            "freqpts": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
        }
        adc_ch = self.cfg.hw.soc.adcs.readout.ch
        xvals = np.arange(self.cfg.expt["expts"])
        phases = 360 * self.cfg.expt["ramsey_freq"] * self.cfg.expt.step * xvals
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(
            self.cfg.expt["expts"]
        )

        for freq in tqdm(freqpts):
            self.cfg.expt.stark_freq = freq
            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for i in range(len(xvals)):
                length = lengths[i]
                phase = phases[i]
                self.cfg.expt.length = float(length)
                self.cfg.expt.phase = float(phase)

                ramsey = RamseyStark2Program(soccfg=self.soccfg, cfg=self.cfg)
                # print(ramsey)
                self.prog = ramsey
                avgi, avgq = ramsey.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["xpts"][-1].append(length)
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)

        data["freqpts"] = freqpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitterfunc = fitter.fitdecaysin
        ydata_lab = ["amps", "avgi", "avgq"]
        ydata_lab = ["avgi"]
        for i, ydata in enumerate(ydata_lab):
            data["fit_" + ydata] = []
            for i in range(len(data["freqpts"])):
                fit_pars = []
                # data['fit_' + ydata], data['fit_err_' + ydata] = fitterfunc(data['xpts'], data[ydata], fitparams=None)
                fit_pars, fit_err, init = fitterfunc(
                    data["xpts"][i], data[ydata][i], fitparams=None
                )
                r2 = fitter.get_r2(data["xpts"], data[ydata], fitter.decaysin, fit_pars)
                fit_err = np.mean(np.abs(fit_err / fit_pars))
                if r2 > 0 and fit_err < 0.5:
                    data["fit_" + ydata].append(fit_pars)
                else:
                    data["fit_" + ydata].append([np.nan] * len(fit_pars))

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        x_sweep = data["xpts"]
        y_sweep = data["freqpts"]
        avgi = data["avgi"]
        avgq = data["avgq"]

        if plot_both:
            fig = plt.figure(figsize=(10, 8))
            plt.subplot(
                211,
                title="Frequency Stark Ramsey Gain" + self.cfg.expt.stark_gain,
                ylabel="Gain [DAC units]",
            )
            plt.pcolormesh(x_sweep, y_sweep, avgi, cmap="viridis", shading="auto")

            plt.colorbar(label="I (ADC level)")
            plt.clim(vmin=None, vmax=None)

            plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Amplitude [MHz]")
            plt.pcolormesh(x_sweep, y_sweep, avgq, cmap="viridis", shading="auto")

            plt.colorbar(label="Q (ADC level)")
            plt.clim(vmin=None, vmax=None)
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.title("Frequency Stark Ramsey")
            plt.ylabel("Gain [DAC units]")
            plt.pcolormesh(x_sweep, y_sweep, avgi, cmap="viridis", shading="auto")

            plt.colorbar(label="I (ADC level)")
            plt.clim(vmin=None, vmax=None)

        plt.tight_layout()
        plt.show()
        if fit:
            plt.figure()
            freq = [data["fit_avgi"][i][1] for i in range(len(data["freqpts"]))]
            plt.plot(data["freqpts"], freq)

        plt.figure(figsize=(10, 6))
        for i in range(len(data["freqpts"])):
            plt.plot(
                data["xpts"][i],
                data["avgi"][i] + 3 * i,
                label=f'Gain {data["freqpts"][i]}',
            )

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


class RamseyStarkPowerExperiment(QickExperiment2DSimple):
    """
    Initialize the T2 Ramsey Stark experiment.
    self.cfg.expt:
        start (float): Wait time tau in microseconds.
        step (float): Step size in microseconds.
        expts (int): Number of experiments.
        start_gain (int): Starting gain value.
        end_gain (int): Ending gain value.
        expts_gain (int): Gain value for experiments.
        ramsey_freq (float): Ramsey frequency in MHz.
        reps (int): Number of repetitions.
        soft_avgs (int): Number of soft_avgs.
        qubit (list): List containing the qubit index.
        stark_freq (float): Stark frequency.
        checkZZ (bool): Flag to check ZZ interaction.
        checkEF (bool): Flag to check EF interaction.
        acStark (bool): Flag to enable AC Stark effect.
        qubit_chan (int): Qubit channel for readout.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        acStark=True,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"ramsey_stark_amp_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        exp_name = RamseyStarkExperiment
        exp_name(cfg_dict, qi, go=False, params=params)

        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 20,
            "start_gain": 0.15,
            "qubit": [qi],
        }
        self.expt = RamseyStarkExperiment(cfg_dict, qi, go=False, params=params, acStark=acStark)
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):

        self.cfg.expt["end_gain"] = np.min(
            [self.cfg.device.qubit.max_gain,self.cfg.expt["end_gain"]])
        gainpts = np.linspace(
            self.cfg.expt["start_gain"],
            self.cfg.expt["end_gain"],
            self.cfg.expt["expts_gain"],
        )

        y_sweep = [{"var": "stark_gain", "pts": gainpts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitterfunc = fitter.fitsin
        super().analyze(fitfunc=fitter.decaysin, fitterfunc=fitterfunc, data=data)


        freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
        popt, pcov = curve_fit(quad_fit, data["stark_gain_pts"], freq)
        data["quad_fit"] = popt

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        title = f"Stark Power Ramsey Q{qubit} Freq: {df}"
        ylabel = "Gain [DAC units]"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=False, title=title, xlabel=xlabel, ylabel=ylabel)

        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
        if fit:
            freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
            ax[0].plot(data["stark_gain_pts"], freq)

            ax[0].plot(
                data["stark_gain_pts"],
                quad_fit(data["stark_gain_pts"], *data["quad_fit"]),
                label="Fit: {:.3g}$x^2$+{:.3g}$x$+{:.3g}".format(*data["quad_fit"]),
            )
            ax[0].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Frequency [MHz]")
            ax[0].legend()
            ax[0].set_title(f"Stark Power Ramsey Q{qubit} Freq: {df}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')

        # Plot raw data
        fig3, ax = plt.subplots(1, 1, figsize=(6, 8))
        for i in range(len(data['stark_gain_pts'])):
            ax.plot(data['xpts'], data['avgi'][i]+18*i)#, label=f'Gain {data['stark_gain_pts'][i]}')

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname

def quad_fit(x, a, b, c):
        return a * x**2 + b * x + c