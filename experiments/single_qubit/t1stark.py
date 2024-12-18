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
from qick_experiment import QickExperimentLoop, QickExperiment2DLoop
import seaborn as sns


class T1StarkProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.soft_avgs = cfg.expt.soft_avgs

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        # self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        # self.r_wait = 3
        # self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(
            cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch
        )
        self.readout_length_dac = self.us2cycles(
            cfg.device.readout.readout_length, gen_ch=self.res_ch
        )
        self.readout_length_adc = self.us2cycles(
            cfg.device.readout.readout_length, ro_ch=self.adc_ch
        )
        self.readout_length_adc += 1  # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = self.adc_ch
        if self.res_ch_type == "int4":
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == "mux4":
            assert self.res_ch == 6
            mask = [0, 1, 2, 3]  # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0] * 4
            mux_freqs[cfg.expt.qubit_chan] = cfg.device.readout.frequency
            mux_gains = [0] * 4
            mux_gains[cfg.expt.qubit_chan] = cfg.device.readout.gain

        self.declare_gen(
            ch=self.res_ch,
            nqz=cfg.hw.soc.dacs.readout.nyquist,
            mixer_freq=mixer_freq,
            mux_freqs=mux_freqs,
            mux_gains=mux_gains,
            ro_ch=ro_ch,
        )

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == "int4":
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(
            ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq
        )

        # declare adcs
        self.declare_readout(
            ch=self.adc_ch,
            length=self.readout_length_adc,
            freq=cfg.device.readout.frequency,
            gen_ch=self.res_ch,
        )

        self.pi_sigma = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch
        )

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == "gauss":
            self.add_gauss(
                ch=self.qubit_ch,
                name="pi_qubit",
                sigma=self.pi_sigma,
                length=self.pi_sigma * 4,
            )
        #    self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        else:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.f_ge,
                phase=0,
                gain=cfg.expt.start,
                length=self.pi_sigma,
            )

        if self.res_ch_type == "mux4":
            self.set_pulse_registers(
                ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask
            )

        else:
            self.set_pulse_registers(
                ch=self.res_ch,
                style="const",
                freq=self.f_res_reg,
                phase=self.deg2reg(-self.cfg.device.readout.phase, gen_ch=self.res_ch),
                gain=cfg.device.readout.gain,
                length=self.readout_length_dac,
            )

        if self.cfg.expt.acStark:
            self.stark_freq = self.freq2reg(cfg.expt.stark_freq, gen_ch=self.qubit_ch)
            self.stark_gain = (
                self.cfg.expt.stark_gain
            )  # gain of the pulse we are trying to calibrate
        self.stark_length = self.us2cycles(cfg.expt.length)

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="arb",
            freq=self.f_ge,
            phase=0,
            gain=cfg.device.qubit.pulses.pi_ge.gain,
            waveform="pi_qubit",
        )
        if self.cfg.expt.do_exp:
            self.pulse(ch=self.qubit_ch)
            # self.sync_all() # align channels
            if self.cfg.expt.acStark:
                self.set_pulse_registers(
                    ch=self.qubit_ch,
                    style="const",
                    freq=self.stark_freq,
                    phase=0,
                    gain=self.stark_gain,  # gain set by update
                    length=self.stark_length,
                )
                self.pulse(ch=self.qubit_ch)
                self.sync_all(5)  # align channels and wait 50ns
            else:
                self.sync_all(self.stark_length)  # align channels and wait 50ns
            # self.wait_all(self.stark_length) # wait for the pulse to finish

        self.measure(
            pulse_ch=self.res_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.device.readout.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.final_delay),
        )

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg = AttrDict(self.cfg)
        shots_i0 = self.di_buf[0] / self.readout_length_adc  # [self.cfg.expt.qubit]
        shots_q0 = self.dq_buf[0] / self.readout_length_adc  # [self.cfg.expt.qubit]
        return shots_i0, shots_q0


class T1StarkExperiment(QickExperimentLoop):
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
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "expts": 60,
            "start": 0.02,
            "span": self.cfg.device.qubit.T1[qi] * 3.7,
            "reps": 2 * self.reps,
            "soft_avgs": self.soft_avgs,
            "stark_gain": 20000,
            "delay_time": self.cfg.device.qubit.T1[qi],
            "df": 70,
            "acStark": True,
            "do_exp": True,
        }
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        params["step"] = params["span"] / (params["expts"] - 1)
        params_exp = {"qubit": [qi], "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi]}
        self.cfg.expt = {**params, **params_exp}

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit[0]
        self.update_config(q_ind)

        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(
            self.cfg.expt["expts"]
        )
        x_sweep = [{"var": "length", "pts": lengths}]
        super().acquire(T1StarkProgram, x_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)

        return self.data

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):

        qubit = self.cfg.expt.qubit
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge
        xlabel = "Wait Time ($\mu$s)"
        captionStr = ["$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"]
        title = f"$T_1$ Stark Q{qubit} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        var = [2]
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
            captionStr=captionStr,
            var=var,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1StarkPowerExperiment(QickExperiment2DLoop):
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
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "expts": 60,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "start": 0.05,
            "start_gain": 0,
            "stop_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 10,
            "acStark": True,
            "qubit": qi,
            "df": 70,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        if style == "fine":
            params_def["soft_avgs"] = params_def["soft_avgs"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        params["step_gain"] = (params["stop_gain"] - params["start_gain"]) / params[
            "expts_gain"
        ]

        params["step"] = params["span"] / params["expts"]
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):

        self.update_config(q_ind=self.cfg.expt.qubit)

        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(
            self.cfg.expt["expts"]
        )
        x_sweep = [{"var": "length", "pts": lengths}]
        gain_pts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * np.arange(
            self.cfg.expt["expts_gain"]
        )
        y_sweep = [{"var": "stark_gain", "pts": gain_pts}]
        self.cfg.expt.do_exp = True
        super().acquire(T1StarkProgram, x_sweep, y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)
        # data['t1_fits']= [self.data['fit_avgi'] ]
        self.data["t1_fits"] = np.array(
            [self.data["fit_avgi"][i][2] for i in range(len(self.data["fit_avgi"]))]
        )

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        ylabel = "Gain [DAC units]"
        title = f"Amplitude Stark T1, Frequency: {self.cfg.expt.stark_freq-self.cfg.device.qubit.f_ge} MHz"
        xlabel = "Wait Time ($\mu$s)"
        super().display(
            data=data,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fit=fit,
            plot_both=plot_both,
        )

        fig2 = plt.figure()
        plt.plot(
            data["stark_gain_pts"] ** 2 / np.max(data["stark_gain_pts"] ** 2),
            data["t1_fits"],
        )
        plt.xlabel("Gain Sq")
        plt.ylabel("T1 (us)")

        imname = self.fname.split("\\")[-1]
        fig2.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "_t1.png"
        )
        plt.show()

        sns.set_palette("coolwarm", n_colors=len(data["t1_fits"]))
        fig3 = plt.figure()
        for i in range(len(data["t1_fits"])):
            plt.plot(
                data["xpts"],
                data["avgi"][i],
                label=f'Gain: {data["stark_gain_pts"][i]}',
            )

        plt.legend()

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class T1StarkPowerContExperiment(QickExperimentLoop):
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


class T1StarkPowerContTimeExperiment(QickExperiment2DLoop):
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


class T1StarkPowerContTime(QickExperiment2DLoop):
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
