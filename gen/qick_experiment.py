import matplotlib.pyplot as plt
import numpy as np
from qick import *
from exp_handling.experiment import Experiment
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import fitting as fitter
import time
import warnings
from scipy.optimize import curve_fit

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

    def acquire(self, prog_name, progress=True, get_hist=True):
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            #final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
            final_delay =10
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

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=True, use_i=None, get_hist=True, **kwargs):
        if data is None:
            data = self.data
        # Remove the last point from fit in case weird edge measurements
        
        # Perform fits on each quadrature
        ydata_lab = ["amps", "avgi", "avgq"]

        if get_hist:
            self.scale_ge()
            ydata_lab.append("scale_data")
        
        for i, ydata in enumerate(ydata_lab):
            (
                data["fit_" + ydata],
                data["fit_err_" + ydata],
                data["fit_init_" + ydata],
            ) = fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], fitparams=None)

        # Get best fit and save error info.
        if use_i is None: 
            use_i = self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]
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
        data['fit_err_par'] = np.sqrt(np.diag(fit_err))/fit_pars
        fit_err = np.mean(np.abs(np.sqrt(np.diag(fit_err)) / fit_pars))
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
        rescale=False,
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
            fig, ax = plt.subplots(3, 1, figsize=(7, 9.5))
            fig.suptitle(title)
            ylabels = ["Amplitude (ADC units)", "I (ADC units)", "Q (ADC units)"]
            ydata_lab = ["amps", "avgi", "avgq"]
        else:
            if ax is None:
                fig, a = plt.subplots(1, 1, figsize=(7, 4))
                ax = [a]
            if rescale: 
                ylabels = ['Excited State Probability']
                ydata_lab = ['scale_data']
            else:
                ylabels = ["I (ADC units)"]
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
                ax[i].legend()
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel(xlabel)
            
            if debug:  # Plot initial guess if debug is True
                pinit = data["init_guess_" + ydata]
                print(pinit)
                ax[i].plot(
                    data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess"
                )

        if show_hist:  # Plot histogram of shots if show_hist is True
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"], "o-")
            try:
                ax.plot(data['bin_centers'], two_gaussians_decay(data['bin_centers'], *data['hist_fit']), label='Fit')
            except:
                pass
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
        hist, bin_edges = np.histogram(shots_i, bins=60, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.1,
        max_err=1,
        disp_kwargs=None,
    ):
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 1
        if disp_kwargs is None:
            disp_kwargs = {}
            # These might be rescale, show_hist, plot_all. Eventually, want to put plot_all into the config. 

        data=self.acquire(progress)
        if analyze:
            data=self.analyze(data)
        if save:
            self.save_data(data)
        if display:
            self.display(data, **disp_kwargs)

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
        # we may want to put these params in the config. 
        params_def = dict(
            threshold_v =self.cfg.device.readout.threshold[qi], 
            read_wait=0.1,
            extra_delay=0.2,
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}
        # this number should be changed to be grabbed from soc 
        self.cfg.expt['threshold']=int(self.cfg.expt['threshold_v']*self.cfg.device.readout.readout_length[qi]/0.0032552083333333335)

    def get_freq(self, fit=True): 
        # Provide correct frequency if mixer's are in use, for two different LO types. 
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

    def scale_ge(self): 
        hist = self.data['hist']
        bin_centers = self.data['bin_centers']
        v_rng = np.max(bin_centers) - np.min(bin_centers)
        
        p0 = [0.5, np.min(bin_centers)+v_rng/3, v_rng/10, np.max(bin_centers)-v_rng/3]
        try:
            popt, pcov = curve_fit(two_gaussians, bin_centers, hist, p0=p0)
            vg = popt[1]
            ve = popt[3]
            if 'tm' in self.cfg.device.readout and self.cfg.device.readout.tm[self.cfg.expt.qubit[0]]!=0:
                tm = self.cfg.device.readout.tm[self.cfg.expt.qubit[0]]
                sigma = self.cfg.device.readout.sigma[self.cfg.expt.qubit[0]]
                p0 = [popt[0], vg, ve]
                popt, pcov = curve_fit(lambda x, mag_g, vg, ve: two_gaussians_decay(x, mag_g, vg, ve, sigma, tm), bin_centers, hist, p0=p0)
                popt = np.concatenate((popt, [sigma, tm]))
            
            dv = popt[2] - popt[1]
            self.data['scale_data'] = (self.data['avgi']-popt[1])/dv
            self.data['hist_fit']=popt
        except:
            self.data['scale_data'] = self.data['avgi']

        
class QickExperimentLoop(QickExperiment):

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=False, qi=0):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, prog_name, x_sweep, progress=True, hist=False):
        
        if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        shots_i =[]
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")
        xvals = np.arange(len(x_sweep[0]["pts"]))
        for i in tqdm(xvals, disable=not progress):
            for j in range(len(x_sweep)):
                self.cfg.expt[x_sweep[j]["var"]] = x_sweep[j]["pts"][i]

            prog = prog_name(soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg)
            
            iq_list = prog.acquire(
                self.im[self.cfg.aliases.soc],
                soft_avgs=self.cfg.expt.soft_avgs,
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            # should add get_params to this to do this properly
            data = self.stow_data(iq_list, data)
            offset = self.soccfg._cfg['readouts'][self.cfg.expt.qubit_chan]["iq_offset"]
            shots_i_new, shots_q = prog.collect_shots(offset=offset)
            shots_i.append(shots_i_new)

            data["xpts"].append(x_sweep[0]["pts"][i])

        bin_centers, hist = self.make_hist(shots_i)
        data["bin_centers"] = bin_centers
        data["hist"] = hist
        for j in range(len(x_sweep)):
            data[x_sweep[j]["var"] + "_pts"] = x_sweep[j]["pts"]
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
        show_hist=True,
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

    def make_hist(self, shots_i):
        hist, bin_edges = np.histogram(shots_i, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist

    def update_config(self, q_ind=None):
        super().update_config(q_ind=q_ind)

    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname


class QickExperiment2D(QickExperimentLoop):

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

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
                
                fit_pars, fit_err, init = fitterfunc(
                    data["xpts"], data[ydata][j], fitparams=None
                )
                data["fit_" + ydata].append(fit_pars)
                data["fit_err_" + ydata].append(fit_err)

        return data

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
            ydata_labs = ["I (ADC level)", "Q (ADC level)"]
            fig.suptitle(title)
        elif plot_amps:
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ydata_labs = ["Amplitude (ADC level)", "Phase (radians)"]
            fig.suptitle(title)
        else:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_title(title)
            ydata_lab = ["avgi"]
            ax = [ax]
            ydata_labs = ["I (ADC level)"]

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


class QickExperiment2DSimple(QickExperiment2D):
    
    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, y_sweep, progress=True):

        data = {"avgi": [], "avgq": [], "amps": [], "phases": [],'xpts':[], 'start_time':[], 'bin_centers':[], 'hist':[]}

        yvals = np.arange(len(y_sweep[0]["pts"]))
        data["time"] = []
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        for i in tqdm(yvals):
            for j in range(len(y_sweep)):
                self.expt.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]
            data_new = self.expt.acquire()
            for key in data_new:
                data[key].append(data_new[key])

        if "count" in [y_sweep[j]["var"] for j in range(len(y_sweep))]:
            data["ypts"] = (data["time"] - np.min(data["time"])) / 3600
        else:
            data["ypts"] = y_sweep[0]["pts"]
        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]

        data['xpts']=data['xpts'][0]
        self.data = data
        return data 
    
    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        data=super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data, fit=fit)
        return data

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


def gaussian(x, mag, cen, wid):
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)

def two_gaussians(x, mag1, cen1, wid, cen2):
    return 1 / np.sqrt(2 * np.pi) / wid * (mag1 *np.exp(-((x - cen1) ** 2) / 2 / wid**2) + (1-mag1) * np.exp(-((x - cen2) ** 2) / 2 / wid**2))

def distfn(v, vg, ve, sigma, tm):
    from scipy.special import erf

    dv = ve - vg
    return np.abs(
        tm
        / 2
        / dv
        * np.exp(tm * (tm * sigma**2 / 2 / dv**2 - (v - vg) / dv))
        * (
            erf((tm * sigma**2 / dv + ve - v) / np.sqrt(2) / sigma)
            + erf((-tm * sigma**2 / dv + v - vg) / np.sqrt(2) / sigma)
        )
    )

def two_gaussians_decay(x, mag_g, vg, ve, sigma, tm):
    yg = gaussian(x, mag_g, vg, sigma)
    ye = gaussian(x, 1-mag_g, ve, sigma) * np.exp(-tm) +(1-mag_g)* distfn(
        x, vg, ve, sigma, tm
    )
    return ye + yg