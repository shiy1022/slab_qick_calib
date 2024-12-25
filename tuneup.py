import autocalib_config as cfg
import experiments as meas
import config
import matplotlib.pyplot as plt
import fitting as fitter
import numpy as np
import scipy.constants as cs
import warnings


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


def measure_temp(cfg_dict, qi, npts=14, reps=None, soft_avgs=None, chan=None):

    rabief = meas.AmplitudeRabiExperiment(cfg_dict, qi=qi, pulse_ge=True, checkEF=True)
    rabief_nopulse = meas.AmplitudeRabiExperiment(
        cfg_dict,
        qi=qi,
        params={"npts": npts},
        pulse_ge=False,
        checkEF=True,
        style="temp",
    )

    # To measure temperature, use fewer points to get more signal more quickly
    fge = 1e6 * rabief.cfg.device.qubit.f_ge[qi]
    if chan is None:
        qubit_temp = (
            1e3
            * cs.h
            * fge
            / (
                cs.k
                * np.log(
                    rabief.data["best_fit"][0] / rabief_nopulse.data["best_fit"][0]
                )
            )
        )
        population = rabief_nopulse.data["best_fit"][0] / rabief.data["best_fit"][0]
    else:
        qubit_temp = (
            1e3
            * cs.h
            * fge
            / (cs.k * np.log(rabief.data[chan][0] / rabief_nopulse.data[chan][0]))
        )
        population = rabief_nopulse.data[chan][0] / rabief.data[chan][0]

    fig, ax = plt.subplots(1, 1)
    i_best = str(rabief.data["i_best"])[2:-1]
    ax[0].plot(
        rabief.data["xpts"],
        rabief.data[i_best] - np.min(rabief.data[i_best]),
        label="ge Pulse",
    )
    ax[0].set_ylabel("Rabei ge Pulse")
    # plt.legend()
    axt = plt.twinx()
    axt.plot(
        rabief_nopulse.data["xpts"],
        rabief_nopulse.data[i_best] - np.min(rabief_nopulse.data[i_best]),
        label="No Pulse",
        color="tab:orange",
    )
    axt.tick_params(axis="y", colors="tab:orange")
    ax.tick_params(axis="y", colors="tab:blue")
    ax.yaxis.label.set_color("tab:blue")
    axt.yaxis.label.set_color("tab:orange")
    axt.set_xlabel("Gain [DAC units]")
    axt.set_ylabel("Amplitude No pulse")
    ax[0].set_title = (
        f"Qubit {qi} Temperature: {qubit_temp:0.2f} mK, Population: {population:0.2g}"
    )

    imname = rabief.fname.split("\\")[-1]
    fig.savefig(rabief.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "temp.png")
    print("Qubit temp [mK]:", qubit_temp)

    print("State preparation ratio:", population)

    print(rabief.data["best_fit"][0])
    print(rabief_nopulse.data["best_fit"][0])

    return qubit_temp, population


def recenter_smart(
    qi, cfg_dict, start="T2r", freq=0.3, max_err=0.45, min_r2=0.1, max_t1=500
):


    # get original frequency
    auto_cfg = config.load(cfg_dict["cfg_file"])
    start_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    freqs = [start_freq]

    # If ramsey is more than 250 kHz away, recenter with qspec. 
    # If it starts with spectroscopy, call find_spec, then run amp_rabi 
    # If fit is failing, try to debug 
    
    params = [
        {"coarse": True, "span": 70},
        {"span": 25},
        {"fine": True},
        {"ramsey_freq": 1.5 * freq, "span": np.pi / np.abs(freq)},
        {},
        {"ramsey_freq": "smart", "npts": 200},
    ]
    freq_error = []
    scan_info = {"scans": scans, "params": params}
    if start == "T2r":
        level = 3
    elif start == "qspec":
        level = 2
    else:
        level = 0

    i = 0
    all_done = False
    ntries = 12
    while i < ntries and not all_done:
        print(f"Level {level}, try {i}")
        prog = run_scan_level(
            qi, cfg_dict, scan_info, level, min_r2=min_r2, max_t1=max_t1
        )
        if prog.status:
            if level > 2:
                freq_error.append(prog.data["f_err"])
                print(f"New f error is {freq_error[-1]:0.3f} MHz")
            if level == 3:
                params[4]["ramsey_freq"] = freq * 0.7
                params[4]["span"] = np.pi / np.abs(freq * 0.7)

            if level == len(scans) - 1:
                all_done = True
            else:
                level += 1

            freqs.append(prog.data["new_freq"])
        else:
            if prog is not None and "new_ramsey_freq" in prog.data:
                params[3]["ramsey_freq"] = prog.data["new_ramsey_freq"]
                params[3]["span"] = np.pi / np.abs(prog.data["new_ramsey_freq"])
                level = 3
            else:
                level -= 1
        i += 1

    print(f"Change in frequency: {freqs[-1]-freqs[-1]:0.3f} MHz")
    auto_cfg = config.load(cfg_dict["cfg_file"])
    end_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    print(f"Qubit {qi} recentered from {start_freq} to {end_freq}")

    if i == ntries:
        return False
    else:
        return True


def run_scan_level(qi, cfg_dict, scan_info, level, min_r2=0.1, max_t1=500):

    if level <= 2:
        soft_avgs = 3
    else:
        soft_avgs = 1
    scan_info["params"][level]["soft_avgs"] = soft_avgs
    prog = scan_info["scans"][level](
        cfg_dict, qi, params=scan_info["params"][level], min_r2=min_r2
    )
    if level > 2 and prog.status:
        freq_arg = np.argmin(np.abs(prog.data["t2r_adjust"]))
        freq = prog.data["t2r_adjust"][freq_arg]
        if np.abs(freq) > 2 * np.abs(prog.cfg.expt.ramsey_freq):
            prog.status = False
            prog.data["new_ramsey_freq"] = 1.3 * freq
    if prog.status:
        config.update_qubit(cfg_dict["cfg_file"], "f_ge", prog.data["new_freq"], qi)

    # if qspec fine, run amp rabi
    if level == 2 and prog.status:
        amp_rabi = meas.AmplitudeRabiExperiment(cfg_dict, qi=qi, min_r2=min_r2)
        if amp_rabi.status:
            config.update_qubit(
                cfg_dict["cfg_file"],
                ("pulses", "pi_ge", "gain"),
                int(amp_rabi.data["pi_length"]),
                qi,
            )
        else:
            prog.status = False
            prog = None

    return prog

    # If you are coming from Ramsey and had signal, try using the frequency you got to recenter iwth
    # If not, use the ramsey with larger frequency
    # First check ramsey with larger frequency - say 2 MHz
    # If that converges, then try with smaller frequency
    # If can't do ramsey, do qubit spec with 8 MHz span low pwoer
    # If not, do it with medium power and 25 MHz
    # If not do it with high power and 100 MHz
    # I think whenever it works, you then call it again.


