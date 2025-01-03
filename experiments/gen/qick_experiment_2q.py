import matplotlib.pyplot as plt
import numpy as np
from qick import *

from exp_handling.experiment import Experiment
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import fitting as fitter
import time
import warnings


class QickExperiment2Q(Experiment):
    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=[0]):

        soccfg = cfg_dict["soc"]
        path = cfg_dict["expt_path"]
        config_file = cfg_dict["cfg_file"]
        im = cfg_dict["im"]
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
            im=im,
        )
        reps = np.max([self.cfg.device.readout.reps[q] for q in qi])
        soft_avgs = np.max([self.cfg.device.readout.soft_avgs[q] for q in qi])
        self.reps = int(
            reps * self.cfg.device.readout.reps_base
        )
        self.soft_avgs = int(
            soft_avgs
            * self.cfg.device.readout.soft_avgs_base
        )

    def acquire(self, prog_name, progress=True, get_hist=True):
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        prog = prog_name(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
        
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            soft_avgs=self.cfg.expt.soft_avgs,
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        xpts = self.get_params(prog)

        amps = np.abs(iq_list[0][0].dot([1, 1j]))
        phases = np.angle(iq_list[0][0].dot([1, 1j]))
        avgi = iq_list[0][0][:, 0]
        avgq = iq_list[0][0][:, 1]

        if get_hist:
            v, hist = self.make_hist(prog)

        data = {
            "xpts": xpts,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "phases": phases,
            "start_time": current_time,
        }
        if get_hist:
            data["bin_centers"] = v
            data["hist"] = hist

        for key in data:
            data[key] = np.array(data[key])
        self.data = data
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, use_i=True, **kwargs):
        if data is None:
            data = self.data
        # Remove the last point from fit in case weird edge measurements

        # Perform fits on each quadrature
        ydata_lab = ["amps", "avgi", "avgq"]
        for i, ydata in enumerate(ydata_lab):
            (
                data["fit_" + ydata],
                data["fit_err_" + ydata],
                data["fit_init_" + ydata],
            ) = fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], fitparams=None)

        # Get best fit and save error info.
        
        if use_i: 
            i_best = "avgi"
            fit_pars = data["fit_avgi"]
            fit_err = data["fit_err_avgi"]
        else:
            fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitfunc)

        r2 = fitter.get_r2(data["xpts"][1:-1], data[i_best][1:-1], fitfunc, fit_pars)
        data["r2"] = r2
        data["best_fit"] = fit_pars
        i_best = i_best.encode("ascii", "ignore")
        data["i_best"] = i_best
        fit_err = np.mean(np.abs(fit_err / fit_pars))
        data["fit_err"] = fit_err
        print(f"R2:{r2:.3f}\tFit par error:{fit_err:.3f}\t Best fit:{i_best}")

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_all=False,
        title="",
        xlabel="",
        fit=True,
        show_hist=False,
        fitfunc=None,
        caption_params=[],
        debug=False,
        **kwargs,
    ):

        if data is None:
            data = self.data

        # If ax is given, can put multiple datasets in one figure
        if ax is None:
            save_fig = True
        else:
            save_fig = False

        # Plot all 3 data sets or just I
        if plot_all:
            fig, ax = plt.subplots(3, 1, figsize=(7.5, 9.5))
            fig.suptitle(title)
            ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
            ydata_lab = ["amps", "avgi", "avgq"]
        else:
            if ax is None:
                fig, a = plt.subplots(1, 1, figsize=(7.5, 4))
                ax = [a]
            ylabels = ["I [ADC units]"]
            ydata_lab = ["avgi"]
            ax[0].set_title(title)

        for i, ydata in enumerate(ydata_lab):
            ax[i].plot(data["xpts"][1:-1], data[ydata][1:-1], "o-")

            if fit:
                p = data["fit_" + ydata]
                pCov = data["fit_err_" + ydata]
                caption = ""
                for j in range(len(caption_params)):
                    if j > 0:
                        caption += "\n"
                    if isinstance(caption_params[j]["index"],int):
                        ind = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=(p[ind]), err=np.sqrt(pCov[ind, ind])
                        )
                    else:
                        var = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=data[var + "_" + ydata]
                        )
                ax[i].plot(
                    data["xpts"][1:-1], fitfunc(data["xpts"][1:-1], *p), label=caption
                )
                ax[i].set_ylabel(ylabels[i])
                ax[i].set_xlabel(xlabel)
                ax[i].legend()
            if debug:  # Plot initial guess if debug is True
                pinit = data["init_guess_" + ydata]
                print(pinit)
                ax[i].plot(
                    data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess"
                )

        if show_hist:  # Plot histogram of shots if show_hist is True
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"]/np.sum(data['hist']), "o-")
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")

        if save_fig:  # Save figure if save_fig is True
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

    def make_hist(self, prog):
        offset = self.soccfg._cfg['readouts'][self.cfg.expt.qubit_chan]["iq_offset"]
        shots_i, shots_q = prog.collect_shots(offset=offset)
        # sturges_bins = int(np.ceil(np.log2(len(shots_i)) + 1))
        hist, bin_edges = np.histogram(shots_i, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.9,
        max_err=0.1,
    ):
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 0.5

        self.go(progress=progress, analyze=analyze, display=display, save=save)
        if (
            "fit_err" in self.data
            and "r2" in self.data
            and self.data["fit_err"] < max_err
            and self.data["r2"] > min_r2
        ):
            self.status = True
        elif "fit_err" not in self.data or "r2" not in self.data:
            pass
        else:
            print("Fit failed")
            self.status = False

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname

    def get_params(self, prog):
        if self.param["param_type"] == "pulse":
            xpts = prog.get_pulse_param(
                self.param["label"], self.param["param"], as_array=True
            )
        else:
            xpts = prog.get_time_param(
                self.param["label"], self.param["param"], as_array=True
            )
        return xpts
    
    def check_params(self, params_def):
        unexpected_params = set(self.cfg.expt.keys()) - set(params_def.keys())
        if unexpected_params:
            warnings.warn(f"Unexpected parameters found in params: {unexpected_params}")


    def configure_reset(self):
        qi = self.cfg.expt.qubit[0]
        params_def = dict(
            threshold_v =self.cfg.device.readout.threshold[qi], 
            read_wait=0.1,
            extra_delay=0.2,
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}
    
        self.cfg.expt['threshold']=int(self.cfg.expt['threshold_v']*self.cfg.device.readout.readout_length[qi]/0.0032552083333333335)


    def get_freq(self, fit=True): 
        freq_offset = 0
        q = self.cfg.expt.qubit[0]
        if "mixer_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.mixer_freq[q]
        # lo_freq is in readout; used for signal core. 
        if "lo_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.lo_freq[q]
        if "lo" in self.cfg.hw.soc and "mixer_freq" in self.cfg.hw.soc.lo:
            freq_offset += self.cfg.hw.soc.lo.mixer_freq[q]
        
        self.data['freq'] = freq_offset + self.data["xpts"]
        self.data["freq_offset"] = freq_offset
        # if fit:
        #     self.data["freq_fit"] = self.data["fit"]
        #     self.data["freq_init"] = self.data["init"]
        #     self.data["freq_fit"][0] = freq_offset + self.data["fit"][0]
        #     self.data["freq_init"][0] = freq_offset + self.data["init"][0]

        
