import experiments as meas
import config
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns
import datetime
colors = ["#0869c8","#b51d14"]

def check_chi(cfg_dict, qi=0, span=7, npts=301, plot=False, check_f=False):

    auto_cfg = config.load(cfg_dict["cfg_file"])
    freq = auto_cfg["device"]["readout"]["frequency"][qi]
    start = freq - 4.5
    center = start + span / 2
    chi = meas.ResSpec(
        cfg_dict,
        qi=qi,
        params={
            "span": span,
            "center": center,
            "npts": npts,
            "soft_avgs": 5,
            "final_delay": 15,
            'pulse_e':True,
        },
        go=False,
    )
    chi.go(analyze=True, display=False, progress=True, save=True)

    if check_f:
        chif = meas.ResSpec(
            cfg_dict,
            qi=qi,
            params={
                "span": span,
                "center": center,
                "npts": npts,
                "soft_avgs": 5,
                "final_delay": 15,
                "pulse_e":True,
                "pulse_f":True,
            },
            go=False,
        )
        chif.go(analyze=True, display=False, progress=True, save=True)

    rspec = meas.ResSpec(
        cfg_dict,
        qi=qi,
        params={
            "span": span,
            "center": center,
            "npts": npts,
            "soft_avgs": 2,
            "final_delay": 15,

        },
        go=False,
    )
    rspec.go(analyze=True, display=False, progress=True, save=True)

    if "mixer_freq" in chi.cfg.hw.soc.dacs.readout:
        xpts_chi = chi.cfg.hw.soc.dacs.readout.mixer_freq + chi.data["xpts"]
        xpts_res = rspec.cfg.hw.soc.dacs.readout.mixer_freq + chi.data["xpts"]
        rspec.data["fit"][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
        chi.data["fit"][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
    else:
        xpts_chi = chi.data["xpts"]
        xpts_res = rspec.data["xpts"]

    arg = np.argmin(np.abs(np.abs(chi.data["amps"])) - np.abs(rspec.data["amps"]))
    arg2 = np.argmin(np.abs(rspec.data["amps"]))
    chi.data["rval"] = rspec.data["xpts"][arg2]
    chi.data["cval"] = chi.data["xpts"][arg]
    chi_val = xpts_chi[arg] - xpts_res[arg2]
    chi.data['chi_val'] = chi_val
    chi.data["freq_opt"] = rspec.data["xpts"][arg]

    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].set_title(f"Chi Measurement Q{qi}")
    ax[0].plot(xpts_res, rspec.data["amps"], label="No Pulse")
    ax[0].plot(xpts_chi, chi.data["amps"], label=f"e Pulse")
    

    cap=f'$\chi=${chi_val:0.2f} MHz'
    ax[0].text(0.04, 0.35, cap, transform=ax[0].transAxes, fontsize=12,
                    verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    ax[0].legend()
    ax[0].axvline(
        x=xpts_chi[arg], color="k", linestyle="--"
    )  # Add vertical line at selected point
    ax[0].axvline(x=xpts_res[arg2], color="k", linestyle="--")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Frequency (MHz)")

    ax[1].plot(xpts_res, rspec.data["amps"] - chi.data["amps"])
    ax[1].axvline(x=xpts_chi[arg], color="k", linestyle="--")
    ax[1].axvline(x=xpts_res[arg2], color="k", linestyle="--")
    ax[1].set_ylabel("Difference")
    ax[1].set_xlabel("Frequency (MHz)")

    ax[2].plot(xpts_res[1:-1], rspec.data["phase_fix"])
    ax[2].plot(xpts_chi[1:-1], chi.data["phase_fix"], label="e Pulse")

    if check_f:
        ax[0].plot(chif.data["xpts"], chif.data["amps"], label="f pulse")
        ax[2].plot(chif.data["xpts"][1:-1], chif.data["phase_fix"], label="f pulse")

    # ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *rspec.data["fit"]),'k')
    # ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *chi.data["fit"]),'k')
    # if check_f:
    #     ax[0].plot(chif.data['xpts'][1:-1], fitter.hangerS21func_sloped(chif.data["xpts"][1:-1], *chif.data["fit"]),'k')

    for a in ax:
        a.set_xlabel("Frequency (MHz)")

    ax[2].set_ylabel("Phase")
    plt.show()

    if check_f:
        ax[0].plot(chif.data["xpts"], chif.data["amps"], label="f pulse")
        ax[2].plot(chif.data["xpts"][1:-1], chif.data["phase_fix"], label="f pulse")

    # ax[0].plot(xpts_res, fitter.hangerS21func_sloped(xpts_res, *rspec.data["fit"]),'k')
    # ax[0].plot(xpts_chi, fitter.hangerS21func_sloped(xpts_chi, *chi.data["fit"]),'k')
    # if check_f:
    #    ax[0].plot(xpts_chi, fitter.hangerS21func_sloped(xpts_chi, *chif.data["fit"]),'k')

    imname = rspec.fname.split("\\")[-1]
    fig.savefig(rspec.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "chi.png")
    return (
        [chi, rspec],
        chi_val,
    )


def measure_temp(cfg_dict, qi, expts=20, soft_avgs=1, chan=None):

    rabief = meas.RabiExperiment(cfg_dict, qi=qi,params={'pulse_ge':True,'checkEF':True})
    rabief_nopulse = meas.RabiExperiment(
        cfg_dict,
        qi=qi,
        params={"expts": expts, 'pulse_ge':False, 'checkEF':True, "soft_avgs": soft_avgs},
        style="temp",
    )

    # To measure temperature, use fewer points to get more signal more quickly
    fge = 1e6 * rabief.cfg.device.qubit.f_ge[qi]
    if chan is None:
        population = rabief_nopulse.data["best_fit"][0] / rabief.data["best_fit"][0]
    else:
        population = rabief_nopulse.data[chan][0] / rabief.data[chan][0]

    qubit_temp = -1e3* cs.h * fge / (cs.k * np.log(population))
    
    fig, ax = plt.subplots(1, 1)
    i_best = str(rabief.data["i_best"])[2:-1]
    ax.plot(
        rabief.data["xpts"],
        rabief.data[i_best] - np.min(rabief.data[i_best]),
        label="ge Pulse",
    )
    ax.set_ylabel("ge Pulse")
    # plt.legend()
    axt = plt.twinx()
    axt.plot(
        rabief_nopulse.data["xpts"],
        rabief_nopulse.data[i_best] - np.min(rabief_nopulse.data[i_best]),
        label="No ge Pulse",
        color=colors[1],
    )
    axt.tick_params(axis="y", colors=colors[1])
    ax.tick_params(axis="y", colors=colors[0])
    ax.yaxis.label.set_color(colors[0])
    axt.yaxis.label.set_color(colors[1])
    axt.set_xlabel("Gain (DAC units)")
    axt.set_ylabel("No ge Pulse")
    ax.set_title(
        f"Qubit {qi} Temperature: {qubit_temp:0.2f} mK, Population: {population:0.2g}"
    )

    imname = rabief.fname.split("\\")[-1]
    fig.savefig(rabief.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "_temp.png")
    print("Qubit temp [mK]:", qubit_temp)

    print("State preparation ratio:", population)

    print(rabief.data["best_fit"][0])
    print(rabief_nopulse.data["best_fit"][0])
    plt.show()

    return qubit_temp, population

def make_hist(d, nbins=200):
    hist, bin_edges = np.histogram(d, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

def plot_reset(d):     
    blue = "#4053d3"
    red = "#b51d14"

    num_plots = len(d)
    fig, ax = plt.subplots( int(np.ceil(num_plots/4)), 4, figsize=(14, 1 * num_plots), sharey=True)
    ax = ax.flatten()

    b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(d[0].data['Igr']))
    for i, shot in enumerate(d):
        v, hist = make_hist(shot.data['Ig'], nbins=50)
        ax[i].semilogy(v, hist, color=blue)
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data['Igr'])):
            v, hist = make_hist(shot.data['Igr'][j],  nbins=50)
            ax[i].semilogy(v, hist,color=b[j])
    
    fig.tight_layout()
    fig, ax = plt.subplots(int(np.ceil(num_plots/4)),4, figsize=(14, 1 * num_plots), sharey=True)
    ax = ax.flatten()
    for i, shot in enumerate(d):
        v, hist = make_hist(shot.data['Ig'], nbins=50)
        ax[i].semilogy(v, hist, color=blue)
        v, hist = make_hist(shot.data['Ie'], nbins=50)
        ax[i].semilogy(v, hist, color=red)
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data['Ier'])):
            v, hist = make_hist(shot.data['Ier'][j], nbins=50)
            ax[i].semilogy(v, hist,color=b[j])

    fig.tight_layout()

    nplots = 6
    fig, ax = plt.subplots(2,nplots, figsize=(nplots*4,8))
    b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(d))
    
    for i, shot in enumerate(d):
        vg, histg = make_hist(shot.data['Ig'], nbins=50)
        ve, histe = make_hist(shot.data['Ie'], nbins=50)
        for j in range(nplots):
            
            ax[0,j].semilogy(vg, histg, color=blue, linewidth=1)
            ax[1,j].semilogy(vg, histg, color=blue, linewidth=1)
            
            ax[1,j].semilogy(ve, histe, color=red, linewidth=1)

            v, hist = make_hist(shot.data['Igr'][j,:], nbins=50)
            ax[0,j].semilogy(v, hist,label=f"{shot.cfg.expt.threshold_v:0.1f}", color=b[i])

            v, hist = make_hist(shot.data['Ier'][j,:], nbins=50)
            ax[1,j].semilogy(v, hist,label=shot.cfg.expt.threshold_v, color=b[i])
            
    ax[0,0].legend(ncol=int(np.ceil(len(d)/6)), fontsize=8)

    ax[0,0].set_title('Ground state')
    ax[1,0].set_title('Excited state')
    fig.tight_layout()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # fig.savefig(
    #             shot.fname[0 : -len(imname)] + "images\\" +  + ".png"
    #         )
    fig.savefig(f"reset_hist_{current_time}.png")

    