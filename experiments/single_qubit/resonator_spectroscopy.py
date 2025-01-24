import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from exp_handling.datamanagement import AttrDict
from scipy.signal import find_peaks
from qick_experiment import QickExperiment, QickExperiment2D
from qick_program import QickProgram
import fitting as fitter
from qick.asm_v2 import QickSweep1D
from scipy.ndimage import gaussian_filter1d
import copy

"""
Measures the resonant frequency of the readout resonator when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readouti.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""


class ResSpecProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        q = cfg.expt.qubit[0]
        self.readout_length = cfg.device.readout.readout_length[q]
        self.phase = cfg.device.readout.phase[q]

        super()._initialize(cfg, readout="")
        self.add_loop("freq_loop", cfg.expt.expts)

        if cfg.expt.pulse_e:
            super().make_pi_pulse(
            cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge"
        )

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            
            if cfg.expt.pulse_f:
                self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
            self.delay_auto(t=0.02, tag="waiting")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
            
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )


class ResSpec(QickExperiment):
    """
    Resonator Spectroscopy Experiment
    Experimental Config
    expt = dict(
        start: start frequency (MHz),
        step: frequency step (MHz),
        expts: number of experiments,
        gain: gain of the readout resonator,
        final_delay: delay time between repetitions in us,
        pulse_e: boolean to add e pulse prior to measurement
        pulse_f: boolean to add f pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        display=True,
        qi=0,
        go=True,
        params={},
        style="fine"
    ):

        prefix = "resonator_spectroscopy_"
        if style == "coarse":
            prefix = prefix + "coarse"
        elif 'pulse_e' in params and params['pulse_e']:
            prefix = prefix + "chi_"
        elif 'pulse_f' in params and params['pulse_f']:
            prefix = prefix + "f_"
        prefix += style + f"_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "gain": self.cfg.device.readout.gain[qi],
            "reps": self.reps,
            "soft_avgs": self.soft_avgs,
            "final_delay": 5,
            "pulse_e": False,
            "pulse_f": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        if style == "coarse":
            params_def["start"] = 6000
            params_def["expts"] = 5000
            params_def["span"] = 500
        else:
            params_def["center"] = self.cfg.device.readout.frequency[qi]
            params_def["expts"] = 200
            params_def["span"] = 5

        # combine params and params_Def, preferring params
        params = {**params_def, **params}

        if params["span"] == "kappa":
            params["span"] = float(8 * self.cfg.device.readout.kappa[qi])
        params = {**params_def, **params}
        if "center" in params:
            params["start"] = params["center"] - params["span"] / 2
        self.cfg.expt = params

        if go:
            if style == "coarse":
                self.go(analyze=False, display=False, progress=True, save=True)
                self.analyze(fit=False, peaks=True)
                self.display(fit=False, peaks=True)
            else:
                super().run(display=display, progress=progress)

    def acquire(self, progress=False):

        self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(ResSpecProgram, progress=progress)

        return self.data

    def analyze(
        self,
        data=None,
        fit=True,
        peaks=False,
        verbose=False,
        hanger=True,
        prom=20,
        debug=False,
        **kwargs,
    ):
        super().get_freq(fit)
        if data is None:
            data = self.data
        
        if fit:
            ydata = data["amps"][1:-1]
            xdata = data["freq"][1:-1]
            fitparams = [
                max(ydata),
                -(max(ydata) - min(ydata)),
                xdata[np.argmin(ydata)],
                0.1,
            ]
            if hanger:
                data["fit"], data["fit_err"], data["init"] = fitter.fithanger(
                    xdata, ydata
                )
                r2 = fitter.get_r2(
                    xdata, ydata, fitter.hangerS21func_sloped, data["fit"]
                )
                data["r2"] = r2
                data["fit_err"] = np.mean(
                    np.sqrt(np.diag(data["fit_err"])) / np.abs(data["fit"])
                )
                if isinstance(data["fit"], (list, np.ndarray)):
                    f0, Qi, Qe, phi, scale, slope = data["fit"]
               
                data["kappa"] = f0 * (1 / Qi + 1 / Qe) * 1e-4
                if verbose:
                    print(
                        f"\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}"
                    )
                    print(f"Freq with maximum transmission: {xdata[np.argmax(ydata)]}")
                    print("From fit:")
                    print(f"\tf0: {f0}")
                    print(f"\tQi: {Qi}")
                    print(f"\tQe: {Qe}")
                    print(f"\tQ0: {1/(1/Qi+1/Qe)}")
                    print(f"\tkappa [MHz]: {f0*(1/Qi+1/Qe)}")
                    print(f"\tphi (radians): {phi}")
                data["freq_fit"]=copy.deepcopy(data["fit"])
                data["freq_init"]=copy.deepcopy(data["init"])
                data["fit"][0]=data["fit"][0]-data["freq_offset"]
                data["init"][0]=data["init"][0]-data["freq_offset"]
            else:
                print(fitparams)
                data["lorentz_fit"] = fitter.fitlor(xdata, ydata, fitparams=fitparams)
                print("From Fit:")
                print(f'\tf0: {data["lorentz_fit"][2]}')
                print(f'\tkappa[MHz]: {data["lorentz_fit"][3]*2}')
        
        data['freq_min']=xdata[np.argmin(ydata)]
        phs_data = np.unwrap(data["phases"][1:-1])
        slope, intercept = np.polyfit(data["xpts"][1:-1], phs_data, 1)
        phs_fix = phs_data - slope * data["xpts"][1:-1] - intercept
        data["phase_fix"] = np.unwrap(phs_fix)
        
        if peaks:
            xdata = data["xpts"][1:-1]
            ydata = data["amps"][1:-1]
            min_dist = 15  # minimum distance between peaks, may need to be edited if things are really close
            max_width = 12  # maximum width of peaks in MHz, may need to be edited if peaks are off
            freq_sigma = 2  # sigma for gaussian filter
            df = xdata[1] - xdata[0]
            min_dist_inds = int(min_dist / df)
            max_width_inds = int(max_width / df)
            filt_sigma = int(freq_sigma / df)
            ydata_smooth = gaussian_filter1d(ydata, sigma=filt_sigma)
            ydata = ydata / ydata_smooth
            if debug:
                fig, ax = plt.subplots(2,1)
                ax[0].plot(xdata, data["amps"][1:-1])
                ax[0].plot(xdata, ydata_smooth)
                ax[1].plot(xdata, ydata)
            coarse_peaks, props = find_peaks(
                -ydata,
                distance=min_dist_inds,
                prominence=prom,
                width=[0, max_width_inds],
            )

            data["coarse_peaks_index"] = coarse_peaks
            data["coarse_peaks"] = xdata[coarse_peaks]
            data["coarse_props"] = props
        return data

    def display(
        self,
        data=None,
        fit=True,
        peaks=False,
        hanger=True,
        debug=False,
        ax=None,
        **kwargs,
    ):
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        qubit = self.cfg.expt.qubit[0]
        title = f"Resonator Spectroscopy Q{qubit}, Gain {self.cfg.expt.gain}"

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(8, 7))
            fig.suptitle(title)
        else:
            ax[0].set_title(title)

        ax[0].set_ylabel("Amps [ADC units]")
        ax[0].plot(data["freq"][1:-1], data["amps"][1:-1], ".-")
        if fit:
            if hanger:
                if not any(np.isnan(data["fit"])):
                    label = f"$\kappa$: {data['kappa']:.2f} MHz"
                    label += f" \n$f$: {data['fit'][0]:.2f} MHz"
                    ax[0].plot(
                        data["freq"],
                        fitter.hangerS21func_sloped(data["freq"], *data["freq_fit"]),
                        label=label,
                    )
                    ax[0].legend()

                if debug:
                    ax[0].plot(
                        data["freq"],
                        fitter.hangerS21func_sloped(data["freq"], *data["freq_init"]),
                        label="Initial fit",
                    )
            elif not any(np.isnan(data["lorentz_fit"])):
                ax[0].plot(
                    data["freq"],
                    fitter.lorfunc(data["lorentz_fit"], data["freq"]),
                    label="Lorentzian fit",
                )
            else:
                print("Lorentzian fit contains NaN values, skipping plot.")
        if peaks:
            num_peaks = len(data["coarse_peaks_index"])
            print("Number of peaks:", num_peaks)
            peak_indices = data["coarse_peaks_index"]
            for i in range(num_peaks):
                peak = peak_indices[i]
                ax[0].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=1)
                ax[1].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=1)

        if savefig:
            ax[1].set_xlabel("Readout Frequency (MHz)")
            ax[1].set_ylabel("Phase (radians)")
            ax[1].plot(data["freq"][1:-1], data["phase_fix"], ".-")
            
            fig.tight_layout()
            plt.show()
            imname = self.fname.split("\\")[-1]
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

        

    def save_data(self, data=None):
        super().save_data(data=data)


class ResSpecPower(QickExperiment2D):
    """
    Keys:
        final_delay (float): Delay time between repetitions in seconds.
        reps (int): Number of repetitions for each experiment.
        soft_avgs (int): Number of soft_avgs for the experiment.
        rng (int): Range for the gain sweep.
        max_gain (int): Maximum gain value.
        expts_gain (int): Number of gain points in the sweep.
        span_f (float): Frequency span for the sweep in MHz.
        expts_f (int): Number of frequency points in the sweep.
        start_gain (int): Starting gain value.
        step_gain (int): Step size for the gain sweep.
        f_off (float): Frequency offset in MHz.
        min_reps (int): Minimum number of repetitions.
        log (bool): Whether to use logarithmic scaling for the gain sweep.
        start_f (float): Starting frequency for the sweep in MHz.
        step_f (float): Step size for the frequency sweep in MHz.
        qubit (int): Qubit index.
        pulse_e (bool): Whether to apply a pulse on the e-channel.
        pulse_f (bool): Whether to apply a pulse on the f-channel.
        pulse_type (str): Type of pulse to apply.
        qubit_chan (int): Qubit channel index.
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        params={},
        pulse_e=False,
    ):
        if pulse_e:
            ef = 'ef_'
        else: 
            ef = ''

        prefix = f"resonator_spectroscopy_power_sweep_{ef}qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": self.reps / 600,
            "soft_avgs": self.soft_avgs,
            "rng": 100,
            "max_gain": self.cfg.device.qubit.max_gain,
            "span": 15,
            "expts": 200,
            "start_gain": 0.003,
            "step_gain": 0.05,
            "expts_gain": 20,
            "f_off": 4,
            "min_reps": 100,
            "final_delay": 5,
            "log": True,
            "qubit": [qi],
            "pulse_e": pulse_e,
            "pulse_f": False,
            "pulse_type": "const",
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        params["start"] = (
            self.cfg.device.readout.frequency[qi]
            - params["span"] / 2
            - params["f_off"]
        )

        self.cfg.expt = params

        if go:
            self.go(analyze=False, display=False, progress=True, save=True)
            self.analyze(fit=True, lowgain=None, highgain=None)
            self.display(fit=True)

    def acquire(self, progress=False):

        if "log" in self.cfg.expt and self.cfg.expt.log == True:
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt["expts_gain"] - 1))

            gain_pts = self.cfg.expt["max_gain"] * rat ** (
                np.arange(self.cfg.expt["expts_gain"])
            )

            rep_list = np.round(
                self.cfg.expt["reps"]
                * (1 / rat ** np.arange(self.cfg.expt["expts_gain"])) ** 2
            )
            rep_list = [int(r) for r in rep_list]
            print(rep_list)
            for i in range(len(rep_list)):
                if rep_list[i] < self.cfg.expt.min_reps:
                    rep_list[i] = self.cfg.expt.min_reps
        else:
            gain_pts = self.cfg.expt["start_gain"] + self.cfg.expt[
                "step_gain"
            ] * np.arange(self.cfg.expt["expts_gain"])
            rep_list = self.cfg.expt["reps"] * np.ones(self.cfg.expt["expts_gain"])
        y_sweep = [{"var": "gain", "pts": gain_pts}, {"var": "reps", "pts": rep_list}]

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(ResSpecProgram, y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data = self.data

        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        if fit:
            if highgain == None:
                highgain = np.max(data["gain_pts"])
            if lowgain == None:
                lowgain = np.min(data["gain_pts"])
            i_highgain = np.argmin(np.abs(data["gain_pts"] - highgain))
            i_lowgain = np.argmin(np.abs(data["gain_pts"] - lowgain))
            fit_highpow, err, pinit = fitter.fitlor(
                data["xpts"], data["amps"][i_highgain]
            )
            fit_lowpow, err, pinitlow = fitter.fitlor(
                data["xpts"], data["amps"][i_lowgain]
            )
            data["fit"] = [fit_highpow, fit_lowpow]
            data["fit_gains"] = [highgain, lowgain]
            data["lamb_shift"] = fit_highpow[2] - fit_lowpow[2]

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        inner_sweep = data[
            "xpts"
        ]  # float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'])
        outer_sweep = data["gain_pts"]

        amps = copy.deepcopy(data["amps"])
        for i in range(len(amps)):
            amps[i, :] = amps[i, :] / np.median(amps[i, :])

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap="viridis", shading="auto")
        if "log" in self.cfg.expt and self.cfg.expt.log:
            plt.yscale("log")
        if fit:
            fit_highpow, fit_lowpow = data["fit"]
            highgain, lowgain = data["fit_gains"]
            plt.axvline(fit_highpow[2], linewidth=1, color="0.2")
            plt.axvline(fit_lowpow[2], linewidth=1, color="0.2")
            plt.plot(x_sweep, [highgain] * len(x_sweep), linewidth=1, color="0.2")
            plt.plot(x_sweep, [lowgain] * len(x_sweep), linewidth=1, color="0.2")
            print(f"High power peak [MHz]: {fit_highpow[2]}")
            print(f"Low power peak [MHz]: {fit_lowpow[2]}")
            print(f'Lamb shift [MHz]: {data["lamb_shift"]}')

        plt.title(f"Resonator Spectroscopy Power Sweep Q{qubit}")
        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Resonator Gain [DAC level]")

        ax.tick_params(
            top=True,
            labeltop=False,
            bottom=True,
            labelbottom=True,
            right=True,
            labelright=False,
        )

        plt.colorbar(label="Amps/Avg (ADC level)")
        plt.show()
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")

    def save_data(self, data=None):
        super().save_data(data=data)


