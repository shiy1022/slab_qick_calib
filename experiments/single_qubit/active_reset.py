import matplotlib.pyplot as plt
import numpy as np
from qick import *
import copy
import seaborn as sns
from exp_handling.datamanagement import AttrDict
from tqdm import tqdm_notebook as tqdm
from gen.qick_experiment import QickExperiment
from gen.qick_program import QickProgram
from scipy.optimize import curve_fit
from scipy.special import erf
from copy import deepcopy

blue = "#4053d3"
red = "#b51d14"
int_rgain = True

# Make it possible to turn fitting off 
def hist(data, plot=True, span=None, ax=None, verbose=False, qubit=0):
    """
    span: histogram limit is the mean +/- span
    """
    # FIXME: f state analysis is broken
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    if "If" in data.keys():
        plot_f = True
        If = data["If"]
        Qf = data["Qf"]
    else:
        plot_f = False

    numbins = 100

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print("Unrotated:")
        print(
            f"Ig {xg:0.3f} +/- {np.std(Ig):0.3f} \t Qg {yg:0.3f} +/- {np.std(Qg):0.3f} \t Amp g {np.abs(xg+1j*yg):0.3f}"
        )
        print(
            f"Ie {xe:0.3f} +/- {np.std(Ie):0.3f} \t Qe {ye:0.3f} +/- {np.std(Qe):0.3f} \t Amp e {np.abs(xe+1j*ye):0.3f}"
        )
        if plot_f:
            print(
                f"If {xf:0.3f} +/- {np.std(If)} \t Qf {yf:0.3f} +/- {np.std(Qf):0.3f} \t Amp f {np.abs(xf+1j*yf):0.3f}"
            )

    """Compute the rotation angle"""
    theta = -np.arctan2((ye - yg), (xe - xg))
    if plot_f:
        theta = -np.arctan2((yf - yg), (xf - xg))

    """Rotate the IQ data"""
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    if plot_f:
        If_new = If * np.cos(theta) - Qf * np.sin(theta)
        Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

    """Post-rotation means of each blob"""
    xg_new, yg_new = np.median(Ig_new), np.median(Qg_new)
    xe_new, ye_new = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print("Rotated:")
        print(
            f"Ig {xg_new:.3f} +/- {np.std(Ig):.3f} \t Qg {yg_new:.3f} +/- {np.std(Qg):.3f} \t Amp g {np.abs(xg_new+1j*yg_new):.3f}"
        )
        print(
            f"Ie {xe_new:.3f} +/- {np.std(Ie):.3f} \t Qe {ye_new:.3f} +/- {np.std(Qe):.3f} \t Amp e {np.abs(xe_new+1j*ye_new):.3f}"
        )
        if plot_f:
            print(
                f"If {xf:.3f} +/- {np.std(If)} \t Qf {yf:.3f} +/- {np.std(Qf):.3f} \t Amp f {np.abs(xf+1j*yf):.3f}"
            )

    if span is None:
        span = (
            np.max(np.concatenate((Ie_new, Ig_new)))
            - np.min(np.concatenate((Ie_new, Ig_new)))
        ) / 2
    xlims = [(xg_new + xe_new) / 2 - span, (xg_new + xe_new) / 2 + span]
    ylims = [yg_new - span, yg_new + span]

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []

    """X and Y ranges for histogram"""

    ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
    ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
    if plot_f:
        nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    tind = contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    err_e = np.cumsum(ne)[tind]/ne.sum()
    err_g = 1-np.cumsum(ng)[tind]/ng.sum()

    if plot_f:
        contrast = np.abs(
            ((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(
            ((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
    m = 0.7
    a = 0.25
    if plot:
        if ax is None:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            ax = [axs[0,1],axs[1,0]]
            
            # Plot unrotated data 
            axs[0, 0].plot(Ig, Qg, ".", label="g", color=blue, alpha=a, markersize=m)
            axs[0, 0].plot(Ie, Qe, ".", label="e", color=red, alpha=a, markersize=m)
            if plot_f:
                axs[0, 0].plot(If, Qf, ".", label="f", color="g", alpha=a, markersize=m)
            axs[0, 0].plot(xg, yg, color="k", marker="o")
            axs[0, 0].plot(xe, ye, color="k", marker="o")
            if plot_f:
                axs[0, 0].plot(xf, yf, color="k", marker="o")

            axs[0,0].set_xlabel('I (ADC levels)')
            axs[0, 0].set_ylabel("Q (ADC levels)")
            axs[0, 0].legend(loc="upper right")
            axs[0, 0].set_title("Unrotated")
            axs[0, 0].axis("equal")
            set_fig=True

            # Plot log histogram         
            bin_cent = (binsg[1:] + binsg[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ng, color=blue)
            bin_cent = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ne, color=red)
            
            axs[1, 1].set_xlabel("I (ADC levels)")
            axs[0, 0].set_xlabel("I (ADC levels)")

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            if qubit is not None: 
                fig.suptitle(f"Single Shot Histogram Analysis Q{qubit}")
        else:
            set_fig=False
        
        # Plot rotated data
        ax[0].plot(Ig_new, Qg_new, ".", label="g", color=blue, alpha=a, markersize=m)
        ax[0].plot(Ie_new, Qe_new, ".", label="e", color=red, alpha=a, markersize=m)
        if plot_f:
            ax[0].plot(If_new, Qf_new, ".", label="f", color="g", alpha=a, markersize=m)
        #ax[0].plot(xg_new, yg_new, color="k", marker="o")
        #ax[0].plot(xe_new, ye_new, color="k", marker="o")
        ax[0].text(0.95, 0.95, f'g: {xg_new:.2f}\ne: {xe_new:.2f}', 
                   transform=ax[0].transAxes, fontsize=10, 
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(facecolor='white', alpha=0.5))

        ax[0].set_xlabel('I (ADC levels)')
        lgnd=ax[0].legend(loc='lower right')
        lgnd.legendHandles[0].set_markersize(6)
        lgnd.legendHandles[1].set_markersize(6)
        ax[0].set_title("Angle: {:.2f}$^\circ$".format(theta * 180 / np.pi))
        ax[0].axis("equal")        

        # Plot histogram 
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("I (ADC levels)")
        ax[1].legend(loc="upper right")
        ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        ax[1].axvline(thresholds[0], color="0.2", linestyle="--")
        ax[1].plot(data["vhg"], data["histg"], '.-',color=blue, markersize=0.5, linewidth=0.3)
        ax[1].fill_between(data["vhg"], data["histg"], color=blue, alpha=0.3)
        ax[1].fill_between(data["vhe"], data["histe"], color=red, alpha=0.3)
        ax[1].plot(data["vhe"], data["histe"], '.-',color=red, markersize=0.5, linewidth=0.3)
        ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
        ax[1].plot(data["vhe"], excited_func(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
        if plot_f:
            nf, binsf, pf = ax[1].hist(
                If_new, bins=numbins, range=xlims, color="g", label="f", alpha=0.5
            )
            ax[1].axvline(thresholds[1], color="0.2", linestyle="--")
            ax[1].axvline(thresholds[2], color="0.2", linestyle="--")

                
        sigma = data['sigma']
        tm = data['tm']
        txt = f"Threshold: {thresholds[0]:.2f}"
        txt += f" \n Width: {sigma:.2f}"
        txt += f" \n $T_m/T_1$: {tm:.2f}"
        ax[1].text(0.025, 0.965, txt, 
               transform=ax[1].transAxes, fontsize=10, 
               verticalalignment='top', horizontalalignment='left', 
               bbox=dict(facecolor='none', edgecolor='black', alpha=0.5))

        if set_fig:
            fig.tight_layout()
        else:
            fig=None
        plt.show()
    else:
        fig = None
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    params = {'fids': fids, 'thresholds': thresholds, 'angle': theta * 180 / np.pi, 'ig':xg_new, 'ie':xe_new, 'err_e':err_e, 'err_g':err_g}
    return params, fig

def gaussian(x, mag, cen, wid):
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)

def two_gaussians(x, mag1, cen1, wid, mag2, cen2):
    return 1 / np.sqrt(2 * np.pi) / wid * (mag1 *np.exp(-((x - cen1) ** 2) / 2 / wid**2) + mag2 * np.exp(-((x - cen2) ** 2) / 2 / wid**2))

def make_hist(d, nbins=200):
    hist, bin_edges = np.histogram(d, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

# Histogram and fit ground state data
def fit_gaussian(d, nbins=200, p0=None, plot=True):
    v, hist = make_hist(d, nbins)
    if p0 is None:
        p0 = [np.mean(v * hist) / np.mean(hist), np.std(d)]
    params, params_err = curve_fit(lambda x, a, b: gaussian(x, 1, a, b), v, hist, p0=p0)
    if plot:
        plt.plot(v, hist, "k.")
        plt.plot(v, gaussian(v, 1, *params), label="g")
    return params, v, hist

# Tail from T1 decay
# vg = ground state voltage, ve = excited state voltage, tm = measurement time/T1 time, sigma = SD of measurement noise
def distfn(v, vg, ve, sigma, tm):
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

# Fit function for excited state
def excited_func(x, vg, ve, sigma, tm):
    y = gaussian(x, 1, ve, sigma) * np.exp(-tm) + distfn(x, vg, ve, sigma, tm)
    return y

# Fit for sum of excited and ground states (adds in fitting of relative magnitudes)
def fit_all(x, mag_g, vg, ve, sigma, tm):
    yg = gaussian(x, mag_g, vg, sigma)
    ye = gaussian(x, 1 - mag_g, ve, sigma) * np.exp(-tm) + (1 - mag_g) * distfn(
        x, vg, ve, sigma, tm
    )
    return ye + yg

def rotate(x, y, theta):
    return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

def full_rotate(d, theta):
    
    d["Ig"], d["Qg"] = rotate(d["Ig"], d["Qg"], theta)
    d["Ie"], d["Qe"] = rotate(d["Ie"], d["Qe"], theta)
    return d

# Fit single shot data
def fit_single_shot(d, plot=True, rot=True):
    # rot: Rotate the data so that signal in I
    data_rot = {}
    d = deepcopy(d)
    if rot:
        params, _ = hist(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data = full_rotate(d, theta)
    else:
        data = d
        theta = 0

    # Fit the 3 gaussian data sets
    paramsg, vhg, histg = fit_gaussian(data["Ig"], nbins=100, p0=None, plot=False)
    paramsqg, vhqg, histqg = fit_gaussian(data["Qg"], nbins=100, p0=None, plot=False)
    paramsqe, vhqe, histqe = fit_gaussian(data["Qe"], nbins=100, p0=None, plot=False)

    vqg = paramsqg[0]
    vqe = paramsqe[0]
    vg = paramsg[0]
    sigma = paramsg[1]

    # Fit the I excited state, including T1 decay.
    # Use previously fit value for vg and sigma
    vhe, histe = make_hist(data["Ie"], nbins=100)
    p0 = [np.mean(vhe * histe) / np.mean(histe), 0.2]
    paramse, params_err = curve_fit(
        lambda x, ve, tm: excited_func(x, vg, ve, sigma, tm), vhe, histe, p0=p0
    )
    ve = paramse[0]
    tm = paramse[1]
    paramse2 = [vg, ve, sigma, tm]
    # Theta from gaussian fit
    theta_corr = -np.arctan2((vqe - vqg), (ve - vg))

    #g_rot = rotate(data["Ig"], data["Qg"], theta_corr)
    #e_rot = rotate(data["Ie"], data["Qe"], theta_corr)

    g_rot = rotate(data['Ig'], data['Qg'], 0)
    e_rot = rotate(data['Ie'], data['Qe'], 0)

    #pg = ['mag', 'cen', 'wid', 'mag', 'cen']
    pg = [0.99, vg, sigma, 0.01, ve]
    #print(pg)
    try: 
        paramsg2, params_err = curve_fit(two_gaussians, vhg, histg, p0=pg)
    except:
        paramsg2 = np.nan
    #print(paramsg2)
    #print(paramsg2-pg)


    data["Ie_rot"] = e_rot[0]
    data["Qe_rot"] = e_rot[1]
    data["Ig_rot"] = g_rot[0]
    data["Qg_rot"] = g_rot[1]

    data["vhg"] = vhg
    data["histg"] = histg
    data["vhe"] = vhe
    data["histe"] = histe

    data["vqg"] = vhqg
    data["histqg"] = histqg
    data["vqe"] = vhqe
    data["histqe"] = histqe

    if plot:
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, excited_func(vhe, vg, ve, sigma, tm), label="e", linewidth=1)
        plt.ylabel("Probability")
        plt.title("Single Shot Histograms")
        plt.xlabel("Voltage")
    p = {'theta': theta, 'vg': vg, 've': ve, 'sigma': sigma, 'tm': tm, 'vqg': vqg, 'vqe': vqe, 'theta_corr': theta_corr}
    return data, p, paramsg, paramse2

# ====================================================== #

class MemoryProgram(QickProgram):

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], pins=[0],t=self.trig_offset)
        self.wait_auto(cfg.expt.read_wait)
        self.read_input(ro_ch=self.adc_ch)
        self.write_dmem(addr=0, src='s_port_l')
        self.write_dmem(addr=1, src='s_port_h')

        if cfg.expt.active_reset:
            self.reset(5)


    def reset(self, i):
        
        # Perform active reset i times 
        cfg = AttrDict(self.cfg)
        for n in range(i):
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
            
            # read the input, test a threshold, and jump if it is met [so, if i<threshold, doesn't do pi pulse]
            self.read_and_jump(ro_ch=self.adc_ch, component='I', threshold=cfg.expt.threshold, test='<', label=f'NOPULSE{n}')
            
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(0.01)
            self.label(f"NOPULSE{n}")

            if n<i-1:
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        
    
    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            #nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]# / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1] #/ nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots

class RepMeasProgram(QickProgram):

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], pins=[0],t=self.trig_offset)

        if cfg.expt.active_reset:
            self.reset(5)


    def reset(self, i):
        
        # Perform active reset i times 
        cfg = AttrDict(self.cfg)
        for n in range(i):
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
            
            if n<i-1:
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                self.delay_auto(0.01)
        
    
    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            #nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]# / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1] #/ nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots


class MemoryExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=False,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):

        if prefix is None:
            prefix = f"single_shot_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            expts=100,
            soft_avgs=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset = False,
            check_e=True,
            check_f=check_f,
            read_wait=0.2,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
        )
        
        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if go:
            self.go(analyze=True, display=False, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if 'setup_reset' in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots

        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        

        ig, qg, ie, qe, g_phase, e_phase, g_norm, e_norm = [], [], [], [], [], [], [], []
        
        for i in range(cfg.expt.expts):
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = False
            cfg.expt.pulse_f = False
            histpro = MemoryProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )
            data["Ig"] = iq_list[0][0][:, 0]
            data["Qg"] = iq_list[0][0][:, 1]
            if self.cfg.expt.active_reset:
                data["Igr"]=iq_list[0][1:,:, 0]

            irawg, qrawg = histpro.collect_shots()
            
            rawd = [irawg[-1], qrawg[-1]]
            # print("buffered readout:", rawd)
            dd = self.soc.read_mem(2,'dmem')
            dd_ang = np.arctan2(dd[1], dd[0])*180/np.pi
            # print("feedback readout:", dd)
            # print("feedback angle:", dd_ang)
            dd_sz = np.sqrt(dd[0]**2 + dd[1]**2)
            # print("g size:", dd_sz)
            ig.append(dd[0])
            qg.append(dd[1])
            g_phase.append(dd_ang)
            g_norm.append(dd_sz)

            # Excited state shots
            if self.cfg.expt.check_e:
                cfg = AttrDict(self.cfg.copy())
                cfg.expt.pulse_e = True
                cfg.expt.pulse_f = False
                histpro = MemoryProgram(
                    soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
                )
                iq_list = histpro.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=progress,
                )

                data["Ie"] = iq_list[0][0][:, 0]
                data["Qe"] = iq_list[0][0][:, 1]
                irawe, qrawe = histpro.collect_shots()
                rawd = [irawe[-1], qrawe[-1]]
                dd = self.soc.read_mem(2,'dmem')
                dd_ang = np.arctan2(dd[1], dd[0])*180/np.pi
                # print("buffered readout:", rawd)
                # print("feedback readout:", dd)
                # print("feedback angle:", dd_ang)
                dd_sz = np.sqrt(dd[0]**2 + dd[1]**2)
                ie.append(dd[0])
                qe.append(dd[1])
                e_phase.append(dd_ang)
                e_norm.append(dd_sz)
                # print("e size:", dd_sz)
                if self.cfg.expt.active_reset:
                    data["Ier"]=iq_list[0][1:,:, 0]
                #print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")
        data = {'ie':ie, 'qe':qe, 'ig':ig, 'qg':qg, 'g_phase':g_phase, 'e_phase':e_phase, 'g_norm':g_norm, 'e_norm':e_norm}
        

        
        keys_list = data.keys()
        for key in keys_list:
            data[key] = np.array(data[key])
        
        mean_data = {key + '_mean': np.mean(data[key]) for key in keys_list}
        std_data = {key + '_std': np.std(data[key]) for key in keys_list}
        
        data.update(mean_data)
        data.update(std_data)

        qi = self.cfg.expt.qubit[0]
        ie=data['ie_mean']/(self.cfg.device.readout.readout_length[qi]/0.0032552083333333335)
        print(ie)
        ig = data['ig_mean']/(self.cfg.device.readout.readout_length[qi]/0.0032552083333333335)
        print(ig)
        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

             
        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data


    def save_data(self, data=None):
        super().save_data(data=data)

    def check_reset(self): 
        nbins=75
        fig, ax = plt.subplots(2,1, figsize=(6,7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = make_hist(self.data['Ig'], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data['Igr']))
        ve, histe = make_hist(self.data['Ie'], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data['Igr'])):
            v, hist = make_hist(self.data['Igr'][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')
            v, hist = make_hist(self.data['Ier'][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind= find_bin_closest_to_value(v, self.data['ie'])
        ind_e= find_bin_closest_to_value(ve, self.data['ie'])
        ind_g= find_bin_closest_to_value(vg, self.data['ie'])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g")

        self.data['reset_e'] = reset_level/e_level
        self.data['reset_g'] = reset_level/g_level



        

        
        ax[0].legend()

        ax[0].set_title('Ground state')
        ax[1].set_title('Excited state')
        plt.show()

class RepMeasExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):

        if prefix is None:
            prefix = f"single_shot_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            soft_avgs=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset = False,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
        )
        
        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if 'setup_reset' in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots
        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        histpro = RepMeasProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]
        if self.cfg.expt.active_reset:
            data["Igr"]=iq_list[0][1:,:, 0]

        irawg, qrawg = histpro.collect_shots()
        
        rawd = [irawg[-1], qrawg[-1]]
        #print("buffered readout:", rawd)

        # Excited state shots
        if self.cfg.expt.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = RepMeasProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )

            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]
            irawe, qraw = histpro.collect_shots()
            #rawd = [iraw[-1], qraw[-1]]
            #print("buffered readout:", rawd)
            #print("feedback readout:", self.soc.read_mem(2,'dmem'))
            if self.cfg.expt.active_reset:
                data["Ier"]=iq_list[0][1:,:, 0]
            #print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")

        # Excited state shots
    

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

        params, _ = hist(
            data=data, plot=False, span=span, verbose=verbose
        )
        data.update(params)
        try:
            data2, p, paramsg, paramse2 = fit_single_shot(data, plot=False)
            data.update(p)
            data["vhg"]=data2["vhg"]
            data["histg"]=data2["histg"]
            data["vhe"]=data2["vhe"]
            data["histe"]=data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
        except:
            print('Fits failed')
             
        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        params, fig = hist(
            data=data, plot=plot, verbose=verbose, span=span, ax=ax, qubit=self.cfg.expt.qubit[0]
        )
        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]
        print(f"ge Fidelity (%): {100*fids[0]:.3f}")
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f"gf Fidelity (%): {100*fids[1]:.3f}")
            print(f"ef Fidelity (%): {100*fids[2]:.3f}")
        print(f"Rotation angle (deg): {angle:.3f}")
        print(f"Threshold ge: {thresholds[0]:.3f}")
        if self.cfg.expt.check_f:
            print(f"Threshold gf: {thresholds[1]:.3f}")
            print(f"Threshold ef: {thresholds[2]:.3f}")
        imname = self.fname.split("\\")[-1]

        if savefig:
            plt.show()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

    def save_data(self, data=None):
        super().save_data(data=data)

    def check_reset(self): 
        nbins=75
        fig, ax = plt.subplots(2,1, figsize=(6,7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = make_hist(self.data['Ig'], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data['Igr']))
        ve, histe = make_hist(self.data['Ie'], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data['Igr'])):
            v, hist = make_hist(self.data['Igr'][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')
            v, hist = make_hist(self.data['Ier'][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind= find_bin_closest_to_value(v, self.data['ie'])
        ind_e= find_bin_closest_to_value(ve, self.data['ie'])
        ind_g= find_bin_closest_to_value(vg, self.data['ie'])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g")

        self.data['reset_e'] = reset_level/e_level
        self.data['reset_g'] = reset_level/g_level



        

        
        ax[0].legend()

        ax[0].set_title('Ground state')
        ax[1].set_title('Excited state')
        plt.show()

# ====================================================== #