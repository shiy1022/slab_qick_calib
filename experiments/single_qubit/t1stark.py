import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss
import time
from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import copy
import fitting as fitter
from qick_experiment import QickExperiment, QickExperiment2DSimple
import seaborn as sns
from qick.asm_v2 import QickSweep1D
import experiments as meas


class T1StarkExperiment(QickExperiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        soft_avgs: number soft_avgs to repeat experiment sweep
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
        style="",
        acStark=True,
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "expts": 60,
            "start": 0.05,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "acStark": acStark,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "stark_gain":1,
            "df": 70,
        }
        params = {**params_def, **params}
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        
        
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if go:
            super().run(min_r2=min_r2, max_err=max_err)


    def acquire(self, progress=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(meas.T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)

        return self.data

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=True):

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Wait Time ($\mu$s)"
        title = f"$T_1$ Stark Q{q} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        fitfunc = fitter.expfunc
        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},           
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

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1StarkPowerExperiment(QickExperiment2DSimple):
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

        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 20,
            "start_gain": 0.15,
            "qubit": [qi],
        }
        self.expt = T1StarkExperiment(cfg_dict, qi, go=False, params=params, acStark=acStark, style=style)
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

        fitterfunc = fitter.fitexp
        super().analyze(fitfunc=fitter.expfunc, fitterfunc=fitterfunc, data=data)

        data['offset'] = [data["fit_avgi"][i][0] for i in range(len(data["stark_gain_pts"]))]
        data['amp'] = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
        data['t1'] = [data["fit_avgi"][i][2] for i in range(len(data["stark_gain_pts"]))]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        title = f"T1 Stark Power Q{qubit} Freq: {df}"
        ylabel = "Gain [DAC units]"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=False, title=title, xlabel=xlabel, ylabel=ylabel)

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))
        
        if fit:           
            ax[0].plot(data["stark_gain_pts"], data['offset'])
            ax[1].plot(data["stark_gain_pts"], data['amp'])
            ax[2].plot(data["stark_gain_pts"], data['t1'])
            
            ax[2].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Offset")
            ax[1].set_ylabel("Amplitude")
            ax[2].set_ylabel("T1")
            ax[0].set_title(f"T1 Stark Power Q{qubit} Freq: {df}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')
        sns.set_palette("coolwarm", len(data["stark_gain_pts"]))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(data['stark_gain_pts'])):
             ax.plot(data['xpts'], data['avgi'][i], linewidth=0.5)#, label=f'Gain {data['stark_gain_pts'][i]}')

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1StarkPowerContExperiment(QickExperiment):
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
        prefix="T1StarkPowerCont",
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
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})

        span_gain = self.cfg.expt.stop_gain - self.cfg.expt.start_gain
        coef = span_gain / np.sqrt(self.cfg.expt["expts"])
        gainpts = self.cfg.expt["start_gain"] + coef * np.sqrt(
            np.arange(self.cfg.expt["expts"])
        )
        data = {
            "xpts": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "avgi_off": [],
            "avgq_off": [],
            "amps_off": [],
            "phases_off": [],
        }

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1

        self.cfg.Eexpt.length = 0
        self.cfg.T1expt.length = self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False

        for gain in tqdm(gainpts):

            self.cfg.expt = copy.deepcopy(self.cfg.Eexpt)
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["avgi_off"].append(avgi)
            data["avgq_off"].append(avgq)
            data["amps_off"].append(amp)
            data["phases_off"].append(phases)

            self.cfg.expt = copy.deepcopy(self.cfg.T1expt)
            self.cfg.expt.stark_gain = gain
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["xpts"].append(gain)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phases)

        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        title = f"Amplitude Stark T1, Frequency: {self.cfg.expt.stark_freq-self.cfg.device.qubit.f_ge} MHz"

        fig = plt.figure(figsize=(7, 8))
        plt.title(title)
        plt.plot(
            data["xpts"] ** 2 / np.max(data["xpts"] ** 2),
            data["avgi"] / data["avgi_off"],
            label="I",
        )
        plt.xlabel("Gain Sq[DAC units]")

        plt.tight_layout()
        plt.show()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


class T1StarkPowerContTimeExperiment(QickExperiment2DSimple):
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
            prefix = f"t1_stark_amp_cont_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "count": 1000,
            "repsT1": 10 * self.reps,
            "repsE": 2 * self.reps,
            "repsG": self.reps,
            "soft_avgs": self.soft_avgs,
            "start_gain": 0,
            "stop_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 200,
            "df": 70,
            "delay_time": self.cfg.device.qubit.T1[qi],
            "start": 0,
            "acStark": acStark,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit[0]
        self.update_config(q_ind)

        span_gain = self.cfg.expt.stop_gain - self.cfg.expt.start_gain
        coef = span_gain / np.sqrt(self.cfg.expt["expts_gain"])
        gainpts = self.cfg.expt["start_gain"] + coef * np.sqrt(
            np.arange(self.cfg.expt["expts_gain"])
        )
        data = {
            "xpts": [],
            "time": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "avgi_e": [],
            "avgq_e": [],
            "amps_e": [],
            "phases_e": [],
            "avgi_g": [],
            "avgq_g": [],
            "amps_g": [],
            "phases_g": [],
        }

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)
        self.cfg.Gexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1
        self.cfg.Gexpt.reps = self.cfg.expt.repsG

        self.cfg.Eexpt.length = 0
        self.cfg.Gexpt.length = 0
        self.cfg.T1expt.length = self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False
        self.cfg.Gexpt.acStark = False
        for tm in tqdm(np.arange(self.cfg.expt.count)):
            data["time"].append(time.time())
            data["avgi_e"].append([])
            data["avgq_e"].append([])
            data["amps_e"].append([])
            data["phases_e"].append([])

            data["avgi_g"].append([])
            data["avgq_g"].append([])
            data["amps_g"].append([])
            data["phases_g"].append([])

            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for gain in gainpts:
                self.cfg.expt = copy.deepcopy(self.cfg.Gexpt)
                self.cfg.expt.do_exp = False
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["avgi_g"][-1].append(avgi)
                data["avgq_g"][-1].append(avgq)
                data["amps_g"][-1].append(amp)
                data["phases_g"][-1].append(phases)

                # Check excited state
                self.cfg.expt = copy.deepcopy(self.cfg.Eexpt)
                self.cfg.expt.do_exp = True

                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["avgi_e"][-1].append(avgi)
                data["avgq_e"][-1].append(avgq)
                data["amps_e"][-1].append(amp)
                data["phases_e"][-1].append(phases)

                self.cfg.expt = copy.deepcopy(self.cfg.T1expt)
                self.cfg.expt.do_exp = True
                self.cfg.expt.stark_gain = gain

                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["xpts"][-1].append(gain)
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phases)

        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)

        data["ypts"] = (data["time"] - np.min(data["time"])) / 3600

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        dv = data["avgi_e"] - data["avgi_g"]
        norm_data = (data["avgi"] - data["avgi_g"]) / dv
        qubit = self.cfg.expt.qubit
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge
        y_sweep = data["ypts"]
        xlabel = "Gain Sq"
        ylabel = "Time (hrs)"
        title = f"$T_1$ Stark Q{qubit[0]} Freq: {df}"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.pcolormesh(
            data["xpts"] ** 2 / np.max(data["xpts"] ** 2), y_sweep, norm_data, label="I"
        )
        plt.colorbar()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1StarkPowerContTime(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment add ground state
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
            prefix = f"t1_stark_amp_cont_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "count": 1000,
            "repsT1": 10 * self.reps,
            "repsE": 2 * self.reps,
            "repsG": self.reps,
            "soft_avgs": self.soft_avgs,
            "expts_f": 200,
            "stop_f": 25,
            "quad_fit_pos": [3e-8, 3e-4, 0],
            "quad_fit_neg": [3e-8, 3e-4, 0],
            "df_pos": 70,
            "df_neg": -70,
            "delay_time": self.cfg.device.qubit.T1[qi],
            "start": 0,
            "acStark": acStark,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        self.cfg.expt = {**params_def, **params}
        self.cfg.expt["stark_freq_pos"] = (
            self.cfg.device.qubit.f_ge[qi] + params["df_pos"]
        )
        self.cfg.expt["stark_freq_neg"] = (
            self.cfg.device.qubit.f_ge[qi] + params["df_neg"]
        )

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit[0]
        self.update_config(q_ind)

        f_pts_pos = np.linspace(0, self.cfg.expt.stop_f, int(self.cfg.expt.expts_f / 2))
        gain_pos = find_inverse_quad_fit(f_pts_pos, *self.cfg.expt.quad_fit_pos)
        f_pts_neg = np.linspace(
            -self.cfg.expt.stop_f, 0, int(self.cfg.expt.expts_f / 2)
        )
        gain_neg = find_inverse_quad_fit(-f_pts_neg, *self.cfg.expt.quad_fit_neg)
        gain_pts = np.concatenate((gain_neg[0:-1], gain_pos))
        f_pts = np.concatenate((f_pts_neg[0:-1], f_pts_pos))

        data = {
            "xpts": [],
            "time": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "avgi_e": [],
            "avgq_e": [],
            "amps_e": [],
            "phases_e": [],
            "avgi_g": [],
            "avgq_g": [],
            "amps_g": [],
            "phases_g": [],
        }

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)
        self.cfg.Gexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1
        self.cfg.Gexpt.reps = self.cfg.expt.repsG

        self.cfg.Eexpt.length = 0
        self.cfg.Gexpt.length = 0
        self.cfg.T1expt.length = self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False
        self.cfg.Gexpt.acStark = False
        for tm in tqdm(np.arange(self.cfg.expt.count)):
            self.cfg.expt = copy.deepcopy(self.cfg.Gexpt)
            self.cfg.expt.do_exp = False
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["avgi_g"].append(avgi)
            data["avgq_g"].append(avgq)
            data["amps_g"].append(amp)
            data["phases_g"].append(phases)

            # Check excited state
            self.cfg.expt = copy.deepcopy(self.cfg.Eexpt)
            self.cfg.expt.do_exp = True

            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )

            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["avgi_e"].append(avgi)
            data["avgq_e"].append(avgq)
            data["amps_e"].append(amp)
            data["phases_e"].append(phases)

            self.cfg.expt = copy.deepcopy(self.cfg.T1expt)
            self.cfg.expt.do_exp = True

            data["time"].append(time.time())
            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for i in range(len(gain_pts)):
                if f_pts[i] < 0:
                    self.cfg.expt.stark_freq = self.cfg.expt.stark_freq_neg
                else:
                    self.cfg.expt.stark_freq = self.cfg.expt.stark_freq_pos

                self.cfg.expt.stark_gain = gain_pts[i]
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["xpts"][-1].append(gain_pts[i])
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phases)

        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)
        data["fpts"] = f_pts
        data["ypts"] = (data["time"] - np.min(data["time"])) / 3600

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        dv = data["avgi_e"] - data["avgi_g"]
        norm_data = (data["avgi"] - data["avgi_g"][:, np.newaxis]) / dv[:, np.newaxis]
        qubit = self.cfg.expt.qubit
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge
        y_sweep = data["ypts"]
        xlabel = "Gain Sq"
        ylabel = "Time (hrs)"
        title = f"$T_1$ Stark Q{qubit[0]} Freq: {df}"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.pcolormesh(data["fpts"], y_sweep, norm_data, label="I")
        plt.colorbar()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


def find_inverse_quad_fit(y, a, b, c):
    rt = []
    for yt in y:
        # Solving the quadratic equation a*x^2 + b*x + (c - y) = 0
        discriminant = b**2 - 4 * a * (c - yt)
        if discriminant < 0:
            return None  # No real roots
        elif discriminant == 0:
            return -b / (2 * a)  # One real root
        else:
            root1 = (-b + np.sqrt(discriminant)) / (2 * a)
            root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        rt.append(root1)
    return rt
