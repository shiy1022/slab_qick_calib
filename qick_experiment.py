import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import fitting as fitter
import time


class QickExperiment(Experiment):
    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):

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
        self.reps = int(
            self.cfg.device.readout.reps[qi] * self.cfg.device.readout.reps_base
        )
        self.soft_avgs = int(
            self.cfg.device.readout.soft_avgs[qi]
            * self.cfg.device.readout.soft_avgs_base
        )

    def acquire(self, prog_name, progress=True, hist=False):
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

        if hist:
            v, hist = self.make_hist(prog)

        data = {
            "xpts": xpts,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "phases": phases,
            "start_time": current_time,
        }
        if hist:
            data["bin_centers"] = v
            data["hist"] = hist
        self.data = data
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        if data is None:
            data = self.data
        # Remove the last point from fit in case weird edge measurements

        ydata_lab = ["amps", "avgi", "avgq"]
        for i, ydata in enumerate(ydata_lab):
            (
                data["fit_" + ydata],
                data["fit_err_" + ydata],
                data["fit_init_" + ydata],
            ) = fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], fitparams=None)

        fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitfunc)
        r2 = fitter.get_r2(data["xpts"][1:-1], data[i_best][1:-1], fitfunc, fit_pars)
        data["r2"] = r2
        data["best_fit"] = fit_pars
        fit_err = np.mean(np.abs(fit_err / fit_pars))
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
        captionStr=[],
        var=[],
        debug=False,
        **kwargs,
    ):

        if data is None:
            data = self.data

        if ax is None:
            savefig = True
        else:
            savefig = False

        if plot_all:
            fig, ax = plt.subplots(3, 1, figsize=(9, 11))
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
                for j in range(len(captionStr)):
                    if j > 0:
                        caption += "\n"
                    caption += captionStr[j].format(
                        val=p[var[j]], err=np.sqrt(pCov[var[j], var[j]])
                    )
                ax[i].plot(
                    data["xpts"][1:-1], fitfunc(data["xpts"][1:-1], *p), label=caption
                )
                ax[i].set_ylabel(ylabels[i])
                ax[i].set_xlabel(xlabel)
                ax[i].legend(loc="upper right")
            if debug:
                pinit = data["init_guess_" + ydata]
                print(pinit)
                ax[i].plot(
                    data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess"
                )

        if show_hist:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"], "o-")

        if savefig:
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

    def update_config(self, q_ind=None):
        if q_ind is None:
            # expand entries in config that are length 1 to fill all qubits
            num_qubits_sample = len(self.cfg.device.qubit.f_ge)
            for subcfg in (
                self.cfg.device.readout,
                self.cfg.device.qubit,
                self.cfg.hw.soc,
            ):
                for key, value in subcfg.items():
                    if isinstance(value, dict):
                        for key2, value2 in value.items():
                            for key3, value3 in value2.items():
                                if not (isinstance(value3, list)):
                                    value2.update({key3: [value3] * num_qubits_sample})
                    elif not (isinstance(value, list)):
                        subcfg.update({key: [value] * num_qubits_sample})
        else:
            for subcfg in (
                self.cfg.device.readout,
                self.cfg.device.qubit,
                self.cfg.hw.soc,
            ):
                for key, value in subcfg.items():
                    if isinstance(value, list):
                        subcfg.update({key: value[q_ind]})
                    elif isinstance(value, dict):
                        for key2, value2 in value.items():
                            for key3, value3 in value2.items():
                                if isinstance(value3, list):
                                    value2.update({key3: value3[q_ind]})

    def make_hist(self, prog):
        shots_i, shots_q = prog.collect_shots()
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


class QickExperimentLoop(QickExperiment):

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=False, qi=0):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, prog_name, x_sweep, progress=True, hist=False):

        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")
        xvals = np.arange(len(x_sweep[0]["pts"]))
        for i in tqdm(xvals, disable=not progress):
            for j in range(len(x_sweep)):
                self.cfg.expt[x_sweep[j]["var"]] = x_sweep[j]["pts"][i]

            prog = prog_name(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            data = self.stow_data(avgi, avgq, data)
            data["xpts"].append(x_sweep[0]["pts"][i])

        for k, a in data.items():
            data[k] = np.array(a)

        data["start_time"] = current_time
        self.data = data

        return data

    def stow_data(self, iq_list, data):
        amps = np.abs(iq_list[0][0].dot([1, 1j]))
        phases = np.angle(iq_list[0][0].dot([1, 1j]))
        avgi = iq_list[0][0][:, 0]
        avgq = iq_list[0][0][:, 1]
        data["avgi"].append(avgi)
        data["avgq"].append(avgq)
        data["amps"].append(amps)
        data["phases"].append(phases)
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False):
        super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data, fit=fit)

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
        captionStr=[],
        var=[],
        debug=False,
    ):
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
            debug=debug,
        )
        pass

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.9,
        max_err=0.1,
    ):
        return super().run(
            progress=progress,
            analyze=analyze,
            display=display,
            save=save,
            min_r2=min_r2,
            max_err=max_err,
        )

    def make_hist(self, prog):
        super().make_hist(prog=prog)

    def update_config(self, q_ind=None):
        super().update_config(q_ind=q_ind)

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class QickExperiment2D(QickExperimentLoop):

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def update_config(self, q_ind=None):
        super().update_config(q_ind=q_ind)

    def acquire(self, prog_name, y_sweep, progress=True):

        data = {"avgi": [], "avgq": [], "amps": [], "phases": []}
        yvals = np.arange(len(y_sweep[0]["pts"]))
        data["time"] = []
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        for i in tqdm(yvals):
            for j in range(len(y_sweep)):
                self.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]
            prog = prog_name(
                soccfg=self.soccfg,
                final_delay=self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(
                self.im[self.cfg.aliases.soc],
                soft_avgs=self.cfg.expt.soft_avgs,
                threshold=None,
                load_pulses=True,
                progress=False,
            )

            data = self.stow_data(iq_list, data)
            data["time"].append(time.time())

        data["xpts"] = self.get_params(prog)
        if "count" in [y_sweep[j]["var"] for j in range(len(y_sweep))]:
            data["ypts"] = (data["time"] - np.min(data["time"])) / 3600
        else:
            data["ypts"] = y_sweep[0]["pts"]
        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]
        for k, a in data.items():
            data[k] = np.array(a)
        data["start_time"] = current_time
        self.data = data
        return data

    def stow_data(self, iq_list, data):
        data = super().stow_data(iq_list, data)
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        if data is None:
            data = self.data
        ydata_lab = ["amps", "avgi", "avgq"]
        ydata_lab = ["avgi"]
        for i, ydata in enumerate(ydata_lab):
            data["fit_" + ydata] = []
            data["fit_err_" + ydata] = []
            for j in range(len(data["ypts"])):
                fit_pars = []
                fit_pars, fit_err, init = fitterfunc(
                    data["xpts"], data[ydata][j], fitparams=None
                )
                data["fit_" + ydata].append(fit_pars)
                data["fit_err_" + ydata].append(fit_err)

    def display(
        self,
        data=None,
        ax=None,
        plot_both=False,
        plot_amps=False,
        title="",
        xlabel="",
        ylabel="",
        **kwargs,
    ):
        if data is None:
            data = self.data
        x_sweep = data["xpts"]
        y_sweep = data["ypts"]

        if ax is None:
            savefig = True
        else:
            savefig = False

        if plot_both:
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["avgi", "avgq"]
            ydata_labs = ["I [ADC level]", "Q [ADC level]"]
            fig.suptitle(title)
        elif plot_amps:
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ydata_labs = ["Amplitude [ADC level]", "Phase [radians]"]
            fig.suptitle(title)
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_title(title)
            ydata_lab = ["avgi"]
            ax = [ax]
            ydata_labs = ["I [ADC level]"]

        for i, ydata in enumerate(ydata_lab):
            ax[i].pcolormesh(
                x_sweep, y_sweep, data[ydata], cmap="viridis", shading="auto"
            )
            plt.colorbar(ax[i].collections[0], ax=ax[i], label=ydata_labs[i])
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

            if "log" in self.cfg.expt and self.cfg.expt.log:
                ax[i].set_yscale("log")

        if savefig:
            fig.tight_layout()
            imname = self.fname.split("\\")[-1]
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.9,
        max_err=0.1,
    ):
        super().run(
            progress=progress,
            analyze=analyze,
            display=display,
            save=save,
            min_r2=min_r2,
            max_err=max_err,
        )

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class QickExperiment2DLoop(QickExperiment2D):

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, prog_name, x_sweep, y_sweep, progress=True):

        xvals = np.arange(len(x_sweep[0]["pts"]))
        yvals = np.arange(len(y_sweep[0]["pts"]))
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        data = {
            "xpts": [],
            "ypts": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
        }
        for k in tqdm(yvals, disable=not progress):
            for j in range(len(y_sweep)):
                var = y_sweep[j]["var"]
                self.cfg.expt[var] = y_sweep[j]["pts"][k]
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])

            for i in tqdm(xvals, disable=True):
                for j in range(len(x_sweep)):
                    var = x_sweep[j]["var"]
                    self.cfg.expt[var] = x_sweep[j]["pts"][i]

                prog = prog_name(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = prog.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )
                data = self.stow_data(avgi, avgq, data)

        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]
        for j in range(len(x_sweep)):
            data[x_sweep[j]["var"] + "_pts"] = x_sweep[j]["pts"]
        data["xpts"] = x_sweep[0]["pts"]
        data["ypts"] = y_sweep[0]["pts"]
        for k, a in data.items():
            data[k] = np.array(a)

        data["start_time"] = current_time
        self.data = data
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data, fit=fit)

    def display(
        self,
        data=None,
        ax=None,
        plot_both=False,
        title="",
        xlabel="",
        ylabel="",
        **kwargs,
    ):
        super().display(
            data=data,
            ax=ax,
            plot_both=plot_both,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def stow_data(self, avgi, avgq, data):
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
        phase = np.angle(avgi + 1j * avgq)  # Calculating the phase
        data["avgi"][-1].append(avgi)
        data["avgq"][-1].append(avgq)
        data["amps"][-1].append(amp)
        data["phases"][-1].append(phase)
        return data

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
