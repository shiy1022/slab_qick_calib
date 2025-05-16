"""
Single Shot Measurement Module

This module provides functionality for single-shot qubit state discrimination and readout optimization.
It includes tools for histogram analysis, fitting of measurement results, and optimization of readout
parameters (frequency, gain, and readout length).
"""

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
import slab_qick_calib.config as config

# Define standard colors for plotting
BLUE = "#4053d3"
RED = "#b51d14"

# ====================================================== #
# Utility Functions for Histogram Analysis and Fitting
# ====================================================== #

def gaussian(x, mag, cen, wid):
    """Gaussian function for fitting."""
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)

def two_gaussians(x, mag1, cen1, wid, mag2, cen2):
    """Sum of two Gaussian functions with shared width."""
    return 1 / np.sqrt(2 * np.pi) / wid * (
        mag1 * np.exp(-((x - cen1) ** 2) / 2 / wid**2) + 
        mag2 * np.exp(-((x - cen2) ** 2) / 2 / wid**2)
    )

def make_hist(d, nbins=200):
    """Create a histogram from data."""
    hist, bin_edges = np.histogram(d, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def fit_gaussian(d, nbins=200, p0=None, plot=True):
    """Fit a Gaussian to data and optionally plot."""
    v, hist = make_hist(d, nbins)
    if p0 is None:
        p0 = [np.mean(v * hist) / np.mean(hist), np.std(d)]
    params, _ = curve_fit(lambda x, a, b: gaussian(x, 1, a, b), v, hist, p0=p0)
    if plot:
        plt.plot(v, hist, "k.")
        plt.plot(v, gaussian(v, 1, *params), label="g")
    return params, v, hist

def distfn(v, vg, ve, sigma, tm):
    """Distribution function for T1 decay during measurement."""
    dv = ve - vg
    return np.abs(
        tm / 2 / dv * np.exp(tm * (tm * sigma**2 / 2 / dv**2 - (v - vg) / dv))
        * (
            erf((tm * sigma**2 / dv + ve - v) / np.sqrt(2) / sigma)
            + erf((-tm * sigma**2 / dv + v - vg) / np.sqrt(2) / sigma)
        )
    )

def excited_func(x, vg, ve, sigma, tm):
    """Fit function for excited state, including T1 decay during measurement."""
    return gaussian(x, 1, ve, sigma) * np.exp(-tm) + distfn(x, vg, ve, sigma, tm)

def fit_all(x, mag_g, vg, ve, sigma, tm):
    """Fit function for sum of excited and ground states with relative magnitudes."""
    yg = gaussian(x, mag_g, vg, sigma)
    ye = gaussian(x, 1 - mag_g, ve, sigma) * np.exp(-tm) + (1 - mag_g) * distfn(
        x, vg, ve, sigma, tm
    )
    return ye + yg

def rotate(x, y, theta):
    """Rotate points in the x-y plane by angle theta."""
    return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

def full_rotate(d, theta):
    """Rotate all IQ data in dictionary by angle theta."""
    d["Ig"], d["Qg"] = rotate(d["Ig"], d["Qg"], theta)
    d["Ie"], d["Qe"] = rotate(d["Ie"], d["Qe"], theta)
    return d

def hist(data, plot=True, span=None, ax=None, verbose=False, qubit=0):
    """Analyze and plot histogram data for qubit state discrimination."""
    # Extract data
    Ig, Qg = data["Ig"], data["Qg"]
    Ie, Qe = data["Ie"], data["Qe"]
    
    # Check if f state data is available
    plot_f = "If" in data
    if plot_f:
        If, Qf = data["If"], data["Qf"]

    numbins = 100

    # Calculate medians for each state
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    # Print unrotated data statistics if verbose
    if verbose:
        print("Unrotated:")
        print(f"Ig {xg:0.3f} +/- {np.std(Ig):0.3f} \t Qg {yg:0.3f} +/- {np.std(Qg):0.3f} \t Amp g {np.abs(xg+1j*yg):0.3f}")
        print(f"Ie {xe:0.3f} +/- {np.std(Ie):0.3f} \t Qe {ye:0.3f} +/- {np.std(Qe):0.3f} \t Amp e {np.abs(xe+1j*ye):0.3f}")
        if plot_f:
            print(f"If {xf:0.3f} +/- {np.std(If)} \t Qf {yf:0.3f} +/- {np.std(Qf):0.3f} \t Amp f {np.abs(xf+1j*yf):0.3f}")

    # Compute rotation angle to maximize separation along I axis
    theta = -np.arctan2((ye - yg), (xe - xg))
    if plot_f:
        theta = -np.arctan2((yf - yg), (xf - xg))

    # Rotate the IQ data
    Ig_new, Qg_new = rotate(Ig, Qg, theta)
    Ie_new, Qe_new = rotate(Ie, Qe, theta)
    if plot_f:
        If_new, Qf_new = rotate(If, Qf, theta)

    # Calculate post-rotation means
    xg_new, yg_new = np.median(Ig_new), np.median(Qg_new)
    xe_new, ye_new = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf_new, yf_new = np.median(If_new), np.median(Qf_new)
    
    # Print rotated data statistics if verbose
    if verbose:
        print("Rotated:")
        print(f"Ig {xg_new:.3f} +/- {np.std(Ig):.3f} \t Qg {yg_new:.3f} +/- {np.std(Qg):.3f} \t Amp g {np.abs(xg_new+1j*yg_new):.3f}")
        print(f"Ie {xe_new:.3f} +/- {np.std(Ie):.3f} \t Qe {ye_new:.3f} +/- {np.std(Qe):.3f} \t Amp e {np.abs(xe_new+1j*ye_new):.3f}")
        if plot_f:
            print(f"If {xf_new:.3f} +/- {np.std(If)} \t Qf {yf_new:.3f} +/- {np.std(Qf):.3f} \t Amp f {np.abs(xf_new+1j*yf_new):.3f}")

    # Set histogram span if not provided
    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new)))) / 2
    xlims = [(xg_new + xe_new) / 2 - span, (xg_new + xe_new) / 2 + span]
    ylims = [yg_new - span, yg_new + span]

    # Compute fidelity using histogram overlap
    fids = []
    thresholds = []

    # Create histograms
    ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
    ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
    if plot_f:
        nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    # Calculate contrast and find optimal threshold
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
    tind = contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    err_e = np.cumsum(ne)[tind]/ne.sum()
    err_g = 1-np.cumsum(ng)[tind]/ng.sum()

    # Calculate additional fidelities for f state if available
    if plot_f:
        for hist_data in [(ng, nf), (ne, nf)]:
            contrast = np.abs(((np.cumsum(hist_data[0]) - np.cumsum(hist_data[1])) / 
                              (0.5 * hist_data[0].sum() + 0.5 * hist_data[1].sum())))
            tind = contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])
    
    # Plot the results if requested
    m = 0.7  # Marker size
    a = 0.25  # Alpha (transparency)
    if plot:
        if ax is None:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            ax = [axs[0,1], axs[1,0]]
            
            # Plot unrotated data 
            axs[0, 0].plot(Ig, Qg, ".", label="g", color=BLUE, alpha=a, markersize=m)
            axs[0, 0].plot(Ie, Qe, ".", label="e", color=RED, alpha=a, markersize=m)
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
            axs[1,1].semilogy(bin_cent, ng, color=BLUE)
            bin_cent = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ne, color=RED)
            axs[1,1].set_title('Log Histogram')
            axs[1, 1].set_xlabel("I (ADC levels)")
            axs[0, 0].set_xlabel("I (ADC levels)")

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            if qubit is not None: 
                fig.suptitle(f"Single Shot Histogram Analysis Q{qubit}")
        else:
            set_fig=False
        
        # Plot rotated data
        ax[0].plot(Ig_new, Qg_new, ".", label="g", color=BLUE, alpha=a, markersize=m)
        ax[0].plot(Ie_new, Qe_new, ".", label="e", color=RED, alpha=a, markersize=m)
        if plot_f:
            ax[0].plot(If_new, Qf_new, ".", label="f", color="g", alpha=a, markersize=m)
        
        # Add text annotation with state centers
        ax[0].text(0.95, 0.95, f'g: {xg_new:.2f}\ne: {xe_new:.2f}', 
                   transform=ax[0].transAxes, fontsize=10, 
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(facecolor='white', alpha=0.5))

        ax[0].set_xlabel('I (ADC levels)')
        lgnd=ax[0].legend(loc='lower right')
        ax[0].set_title(f"Angle: {theta * 180 / np.pi:.2f}$^\circ$")
        ax[0].axis("equal")        

        # Plot histogram 
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("I (ADC levels)")
        ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        ax[1].axvline(thresholds[0], color="0.2", linestyle="--")
        ax[1].plot(data["vhg"], data["histg"], '.-', color=BLUE, markersize=0.5, linewidth=0.3)
        ax[1].fill_between(data["vhg"], data["histg"], color=BLUE, alpha=0.3)
        ax[1].fill_between(data["vhe"], data["histe"], color=RED, alpha=0.3)
        ax[1].plot(data["vhe"], data["histe"], '.-', color=RED, markersize=0.5, linewidth=0.3)
        ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
        ax[1].plot(data["vhe"], excited_func(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
        
        if plot_f:
            nf, binsf, pf = ax[1].hist(
                If_new, bins=numbins, range=xlims, color="g", label="f", alpha=0.5
            )
            ax[1].axvline(thresholds[1], color="0.2", linestyle="--")
            ax[1].axvline(thresholds[2], color="0.2", linestyle="--")

        # Add text annotation with fit parameters
        sigma = data['sigma']
        tm = data['tm']
        txt = f"Threshold: {thresholds[0]:.2f}\nWidth: {sigma:.2f}\n$T_m/T_1$: {tm:.2f}"
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

    # Return parameters
    params = {
        'fids': fids, 
        'thresholds': thresholds, 
        'angle': theta * 180 / np.pi, 
        'ig': xg_new, 
        'ie': xe_new, 
        'err_e': err_e, 
        'err_g': err_g
    }
    return params, fig

def fit_single_shot(d, plot=True, rot=True):
    """Fit single shot data with Gaussian and T1 decay models."""
    # Make a copy of the data to avoid modifying the original
    d = deepcopy(d)
    
    # Rotate data if requested
    if rot:
        params, _ = hist(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data = full_rotate(d, theta)
    else:
        data = d
        theta = 0

    # Fit the ground state data
    paramsg, vhg, histg = fit_gaussian(data["Ig"], nbins=100, p0=None, plot=False)
    paramsqg, vhqg, histqg = fit_gaussian(data["Qg"], nbins=100, p0=None, plot=False)
    paramsqe, vhqe, histqe = fit_gaussian(data["Qe"], nbins=100, p0=None, plot=False)

    vqg = paramsqg[0]
    vqe = paramsqe[0]
    vg = paramsg[0]
    sigma = paramsg[1]

    # Fit the excited state data, including T1 decay
    vhe, histe = make_hist(data["Ie"], nbins=100)
    p0 = [np.mean(vhe * histe) / np.mean(histe), 0.2]
    paramse, _ = curve_fit(
        lambda x, ve, tm: excited_func(x, vg, ve, sigma, tm), vhe, histe, p0=p0
    )
    ve = paramse[0]
    tm = paramse[1]
    paramse2 = [vg, ve, sigma, tm]
    
    # Calculate correction angle from Gaussian fit
    theta_corr = -np.arctan2((vqe - vqg), (ve - vg))

    # Rotate data for final analysis
    g_rot = rotate(data['Ig'], data['Qg'], 0)
    e_rot = rotate(data['Ie'], data['Qe'], 0)

    # Try to fit two Gaussians to ground state data (for potential leakage)
    pg = [0.99, vg, sigma, 0.01, ve]
    try: 
        paramsg2, _ = curve_fit(two_gaussians, vhg, histg, p0=pg)
    except:
        paramsg2 = np.nan

    # Store rotated data and histograms
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

    # Plot the fits if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, excited_func(vhe, vg, ve, sigma, tm), label="e", linewidth=1)
        plt.ylabel("Probability")
        plt.title("Single Shot Histograms")
        plt.xlabel("Voltage")
        plt.legend()
    
    # Return results
    p = {
        'theta': theta, 
        'vg': vg, 
        've': ve, 
        'sigma': sigma, 
        'tm': tm, 
        'vqg': vqg, 
        'vqe': vqe, 
        'theta_corr': theta_corr
    }
    return data, p, paramsg, paramse2

# ====================================================== #
# QICK Program for Single Shot Measurements
# ====================================================== #

class HistogramProgram(QickProgram):
    """QICK program for single shot measurements."""

    def __init__(self, soccfg, final_delay, cfg):
        """Initialize the program."""
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """Initialize program parameters and pulses."""
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        # Set up readout parameters
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]] if cfg.expt.active_reset else 0
        self.readout_length = cfg.expt.readout_length
        
        # Initialize parent class
        super()._initialize(cfg, readout="")

        # Create pi pulses for state preparation
        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        if cfg.expt.pulse_f:
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")
        
        # Give the tProc some time for initial setup
        self.delay(0.5)

    def _body(self, cfg):
        """Define the main body of the program."""
        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        # Apply state preparation pulses if requested
        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        
        # Wait before readout
        self.delay_auto(t=0.01, tag="wait")

        # Apply readout pulse and trigger acquisition
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)

        # Apply active reset if configured
        if cfg.expt.active_reset:
            self.reset(7)

    def reset(self, i):
        """Reset method for active reset."""
        super().reset(i)
    
    def collect_shots(self, offset=0):
        """Collect and process shot data."""
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0].flatten()
            q_shots = iq_raw[i][:, :, 0, 1].flatten()
        return i_shots, q_shots

# ====================================================== #
# Experiment Classes
# ====================================================== #

class HistogramExperiment(QickExperiment):
    """Single Shot Histogram Experiment."""

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
        """Initialize the experiment."""
        if prefix is None:
            prefix = f"single_shot_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Set default parameters
        params_def = dict(
            shots=10000,
            reps=1,
            soft_avgs=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset=False,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
            ddr4=False,
        )
        
        # Update with user-provided parameters
        self.cfg.expt = {**params_def, **params}
        
        # Configure reset if needed
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        # Run the experiment if requested
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):
        """Acquire data for the experiment."""
        data = dict()
        
        # Determine final delay based on configuration
        if 'setup_reset' in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots
        cfg = AttrDict(copy.deepcopy(dict(self.cfg)))
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        # Create and configure the histogram program
        histpro = HistogramProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)

        # Configure DDR4 if needed
        if self.cfg.expt.ddr4:
            n_transfers = 1500000  # each transfer (aka burst) is 256 decimated samples
            nt = n_transfers
            # Arm the buffers
            self.im[self.cfg.aliases.soc].arm_ddr4(ch=self.cfg.expt.qubit_chan, nt=n_transfers)

        # Acquire ground state data
        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        
        # Store ground state data
        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]
        if self.cfg.expt.active_reset:
            data["Igr"] = iq_list[0][1:, :, 0]
        
        # Get DDR4 data if configured
        if self.cfg.expt.ddr4:
            iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
            t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
            data['t_g'] = t
            data['iq_ddr4_g'] = iq_ddr4
        
        # Collect raw shots
        irawg, qrawg = histpro.collect_shots()

        # Excited state shots
        if self.cfg.expt.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            
            # Configure DDR4 if needed
            if self.cfg.expt.ddr4:
                self.im[self.cfg.aliases.soc].arm_ddr4(ch=self.cfg.expt.qubit_chan, nt=n_transfers)
            
            # Acquire excited state data
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )
            
            # Get DDR4 data if configured
            if self.cfg.expt.ddr4:
                iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
                t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
                data['t_e'] = t
                data['iq_ddr4_e'] = iq_ddr4

            # Store excited state data
            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]
            
            if self.cfg.expt.active_reset:
                data["Ier"] = iq_list[0][1:, :, 0]

        # F state shots (if requested)
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
