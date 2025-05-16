# Utility functions for data analysis and fitting
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
blue = "#4053d3"
red = "#b51d14"
from copy import deepcopy

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
            axs[1,1].set_title('Log Histogram')
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
        # lgnd.legendHandles[0].set_markersize(6)
        # lgnd.legendHandles[1].set_markersize(6)
        ax[0].set_title("Angle: {:.2f}$^\circ$".format(theta * 180 / np.pi))
        ax[0].axis("equal")        

        # Plot histogram 
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("I (ADC levels)")
        #ax[1].legend(loc="upper right")
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
        paramsg2, params_err = curve_fit(two_gaussians, vhg, histg, p0=pg) #@IgnoreException
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