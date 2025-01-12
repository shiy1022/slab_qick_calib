import experiments as meas
import config
import matplotlib.pyplot as plt
import fitting as fitter
import numpy as np
import scipy.constants as cs
import warnings
from datetime import datetime
import tuneup 
max_t1=500
max_err = 1
min_r2 = 0.35
tol=0.3
plt.rcParams['legend.handlelength'] = 0.5

def tune_up_qubit(qi, cfg_dict, update=True, first_time=False, readout=True, single=False, max_t1=500, max_err=1, min_r2=0.35, tol=0.3):
        cfg_path = cfg_dict['cfg_file']
        auto_cfg = config.load(cfg_path)
        
        # Resonator spectroscopy 
        rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa'})
        if readout and rspec.status: 
            auto_cfg = config.update_readout(cfg_path, 'frequency', rspec.data['fit'][0], qi)
        if update and rspec.status:
            auto_cfg = config.update_readout(cfg_path, 'qi', rspec.data['fit'][1], qi)
            auto_cfg = config.update_readout(cfg_path, 'qe', rspec.data['fit'][2], qi)
            auto_cfg = config.update_readout(cfg_path, 'kappa', rspec.data['kappa'], qi, rng_vals=[0.03, 10])

        if not first_time: 
            shot=meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
            if update:
                config.update_readout(cfg_path, 'phase', shot.data['angle'], qi);
                config.update_readout(cfg_path, 'threshold', shot.data['thresholds'][0], qi);
                config.update_readout(cfg_path, 'fidelity', shot.data['fids'][0], qi);
                config.update_readout(cfg_path, 'sigma', shot.data['sigma'], qi);
                config.update_readout(cfg_path, 'tm', shot.data['tm'],qi);
                if shot.data['fids'][0]>0.06:
                    config.update_qubit(cfg_path, 'tuned_up', True, qi);
        
        # Fine qubit spectroscopy
        qspec=meas.QubitSpec(cfg_dict, qi=qi, style='fine', params={'span':3,'expts':85,'soft_avgs':2, 'length':'t1'})     
        if not qspec.status:
            find_spec(qi, cfg_dict, start="medium")

        if np.abs(qspec.data['new_freq']-auto_cfg.device.qubit.f_ge[qi])>0.25 and qspec.status:
            print('Qubit frequency is off spectroscopy by more than 250 kHz, recentering')
            auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data['new_freq'], qi)
        
        # Amp Rabi 
        amp_rabi = meas.RabiExperiment(cfg_dict,qi=qi)
        if update:
            if amp_rabi.status:
                config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)
                

        
        # Run T1 to get sense of coherence times
        if first_time:
            t1=meas.T1Experiment(cfg_dict, qi=qi)
            if update and t1.status: 
                auto_cfg = config.update_qubit(cfg_path, 'T1', t1.data['new_t1_i'], qi,sig=2, rng_vals=[1.5, max_t1*2])
                auto_cfg = config.update_readout(cfg_path, 'final_delay', 6*t1.data['new_t1'], qi, sig=2,rng_vals=[10, 1000])
                if first_time:
                    auto_cfg = config.update_qubit(cfg_path, 'T2r', t1.data['new_t1_i'], qi,sig=2, rng_vals=[1.5, max_t1*2])
                    auto_cfg = config.update_qubit(cfg_path, 'T2e', 2*t1.data['new_t1'], qi,sig=2, rng_vals=[1.5, max_t1*2])

            # Run single shot opt to improve readout 
            shot=meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
            if update:
                config.update_readout(cfg_path, 'phase', shot.data['angle'], qi);
                config.update_readout(cfg_path, 'threshold', shot.data['thresholds'][0], qi);
                config.update_readout(cfg_path, 'fidelity', shot.data['fids'][0], qi);
                config.update_readout(cfg_path, 'sigma', shot.data['sigma'], qi);
                config.update_readout(cfg_path, 'tm', shot.data['tm'],qi);
                if shot.data['fids'][0]>0.06:
                    config.update_qubit(cfg_path, 'tuned_up', True, qi);

        # Run Ramsey first to center
        if first_time: 
            recenter(qi, cfg_dict, style='coarse')
        else:
            recenter(qi, cfg_dict, style='fine')

        # Amp rabi to improve pi pulse
        amp_rabi = meas.RabiExperiment(cfg_dict,qi=qi)
        if update:
            if amp_rabi.status:
                config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), amp_rabi.data['pi_length'], qi)

        # Run SS Opt fine and SS 
        if single: 
            shotopt=meas.SingleShotOptExperiment(cfg_dict, qi=qi,params={'expts_f':1, 'expts_gain':5, 'expts_len':5},style='fine')
            if update: 
                config.update_readout(cfg_path, 'gain', shotopt.data['gain'], qi);
                config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi);

        shot=meas.HistogramExperiment(cfg_dict, qi=qi, params={'shots':20000})
        if update:
            config.update_readout(cfg_path, 'phase', round(float(shot.data['angle']),3), qi);
            config.update_readout(cfg_path, 'threshold', round(float(shot.data['thresholds'][0]),4), qi);
            config.update_readout(cfg_path, 'fidelity', round(float(shot.data['fids'][0]),4), qi);
            config.update_readout(cfg_path, 'sigma', shot.data['sigma'], qi);
            config.update_readout(cfg_path, 'tm', shot.data['tm'],qi);

        # Once readout tuned and qubit centered, get all the coherences. 
        # Run Ramsey for coherence 
        t2r= get_coherence(meas.RamseyExperiment, qi, cfg_dict,par='T2r')

        # Run T1 
        t1= get_coherence(meas.T1Experiment, qi, cfg_dict,par='T1')

        # Run T2 Echo 
        t2= get_coherence(meas.RamseyEchoExperiment, qi, cfg_dict,par='T2e')
        #t2e= get_coherence(meas.RamseyEchoExperiment, qi, cfg_dict,par='T2e')

        # Run chi 
        chid, chi_val=tuneup.check_chi(cfg_dict, qi=qi)
        auto_cfg = config.update_readout(cfg_path, 'chi', float(chi_val), qi)
        progs = {'amp_rabi':amp_rabi, 't1':t1, 't2r':t2r, 't2e':t2e, 'shot':shot, 'rspec':rspec, 'qspec':qspec, 'chid':chid}
        #progs = {'amp_rabi':amp_rabi, 't1':t1, 't2r':t2r, 'shot':shot, 'rspec':rspec, 'qspec':qspec, 'chid':chid}
        make_summary_figure(cfg_dict, progs, qi)


def make_summary_figure(cfg_dict, progs, qi):    

    auto_cfg = config.load(cfg_dict['cfg_file'])
    fig, ax = plt.subplots(3,3, figsize=(14,9))
    progs['amp_rabi'].display(ax=[ax[0,0]])
    progs['t1'].display(ax=[ax[0,1]])
    progs['t2r'].display(ax=[ax[1,0]])
    progs['t2e'].display(ax=[ax[1,1]])
    progs['shot'].display(ax=[ax[0,2],ax[1,2]])
    progs['rspec'].display(ax=[ax[2,0]])

    
    
    cap = f'Length: {auto_cfg.device.readout.readout_length[qi]:0.2f} $\mu$s'
    cap += f'\nGain: {auto_cfg.device.readout.gain[qi]}'
    ax[0,2].text(0.02, 0.05, cap, transform=ax[0,2].transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))

    chi_fig(ax[2,1], qi, progs)
    
    progs['qspec'].display(ax=[ax[2,2]], plot_all=False)
    

    ax[2,2].set_title(f'Qubit Freq: {auto_cfg.device.qubit.f_ge[qi]:0.2f} MHz')
    ax[2,2].axvline(x=auto_cfg.device.qubit.f_ge[qi], color='k', linestyle='--')

    # ax[2,1].set_title(f'Chi Measurement Q{qi}')
    # ax[2,1].plot(progs['chid'][1].data['xpts'], progs['chid'][1].data['amps'], label='No Pulse')
    # ax[2,1].plot(progs['chid'][0].data['xpts'], progs['chid'][0].data['amps'], label=f'e Pulse')
    
    # chi_val = progs['chid'][0].data['chi_val']
    # cap=f'$\chi=${chi_val:0.2f} MHz'
    # ax[2,1].text(0.04, 0.35, cap, transform=ax[2,1].transAxes, fontsize=10,
    #                 verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    # ax[2,1].axvline(x=progs['chid'][0].data['cval'], color='k', linestyle='--')  # Add vertical line at selected point
    # ax[2,1].axvline(x=progs['chid'][0].data['rval'], color='k', linestyle='--')
    # ax[2,1].legend()
    # ax[2,1].set_ylabel('Amplitude')
    # ax[2,1].set_xlabel('Frequency (MHz)')
    
    fig.tight_layout()
    plt.show()
    
    datestr = datetime.now().strftime("%Y%m%d_%H%M")
    fname = cfg_dict['expt_path'] + f'images\\summary\\qubit{qi}_tuneup_{datestr}.png'
    print(fname)
    fig.savefig(fname)


def chi_fig(ax, qi, progs):
    ax.set_title(f'Chi Measurement Q{qi}')
    ax.plot(progs['chid'][1].data['xpts'], progs['chid'][1].data['amps'], label='No Pulse')
    ax.plot(progs['chid'][0].data['xpts'], progs['chid'][0].data['amps'], label=f'e Pulse')
    
    chi_val = progs['chid'][0].data['chi_val']
    cap=f'$\chi=${chi_val:0.2f} MHz'
    ax.text(0.04, 0.35, cap, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    ax.axvline(x=progs['chid'][0].data['cval'], color='k', linestyle='--')  # Add vertical line at selected point
    ax.axvline(x=progs['chid'][0].data['rval'], color='k', linestyle='--')
    ax.legend()
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency (MHz)')

def find_spec(qi, cfg_dict, start="coarse", freq='ge', max_err=0.45, min_r2=0.1):

    if freq == 'ef':
        f = 'f_ef'
        params = {'checkEF':True}
    else:
        f = 'f_ge'
        params = {}


    style = ["huge","coarse", "medium", "fine"]
    level = style.index(start)

    i = 0
    all_done = False
    ntries = 6
    while i < ntries and not all_done:        
        print(level)
        prog = meas.QubitSpec(cfg_dict, qi=qi, min_r2=min_r2, params=params, style=style[level])
        if prog.status:
            config.update_qubit(cfg_dict["cfg_file"], f, prog.data["new_freq"], qi)
        if prog.status:
            if level == len(style) - 1:
                all_done = True
                print(f'Found qubit {qi}')
            else:
                level += 1
                #params[level]["center"] = prog.data["new_freq"]
        else:
            level -= 1
        if level<0:
            print('Coarsest scan failed, adding more power and reps')
            auto_cfg = config.load(cfg_dict["cfg_file"])
            # does this work?
            auto_cfg.device.qubit.spec_gain[qi] = auto_cfg.device.qubit.spec_gain[qi]*2 
            auto_cfg.device.readout.reps[qi] = auto_cfg.device.readout.reps[qi]*2
            level=0
        i += 1

    if i == ntries:
        return False, i
    else:
        return True, i
    

def get_coherence(
    scan_name,
    qi=0,
    cfg_dict={},
    par="T1",
    params=None,
    min_r2=0.1,
    max_err=0.5,
    tol=0.3,
    max_t1=500,
):
    # For t1, t2r, t2e
    if params is None: 
        params = {}
    err = 2 * tol
    print(params)
    auto_cfg = config.load(cfg_dict["cfg_file"])
    old_par = auto_cfg["device"]["qubit"][par][qi]
    i = 0
    while err > tol and i < 5:
        prog = scan_name(cfg_dict, qi=qi, params=params)
        if par == "T1":
            new_par = prog.data["new_t1_i"]
        else:
            if "best_fit" in prog.data:
                new_par = prog.data["fit_avgi"][3]
        if prog.status:
            auto_cfg = config.update_qubit(
                cfg_dict["cfg_file"], par, new_par, qi, sig=2, rng_vals=[1.5, max_t1]
            )
            err = np.abs(new_par - old_par) / old_par
        elif prog.data["fit_err"] > max_err:
            print("Fit Error too high")
            params["span"] = 2 * old_par * 3 # Usually occurs because too little signal 
            err = 2 * tol
        else:
            print("Failed")
            if 'soft_avgs' in params:
                params['soft_avgs'] = 2*params['soft_avgs']
            else:
                params['soft_avgs'] = 2
            err = 2 * tol
            print('Increasing soft avgs due to fitting issues')

        old_par = new_par
        i += 1

    return prog

def recenter(
    qi, cfg_dict, max_err=0.5, min_r2=0.1, max_t1=500, style='coarse',
):

    # get original frequency
    auto_cfg = config.load(cfg_dict["cfg_file"])
    start_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    freqs = [start_freq]
    if style=='coarse':
        freq = 0.2 
    else:
        freq = 0.2
    params = {'ramsey_freq':freq}
    freq_error = []

    i = 0
    err = 1 
    tol = 0.02
    ntries = 3
    while i < ntries and err>tol :
        print(f"Try {i}")
        prog = meas.RamseyExperiment(cfg_dict, qi=qi, params=params)
        if prog.status:
            freq_error.append(prog.data["f_err"])
            err = np.abs(freq_error[-1])
            print(f"Scan successful. New f error is {freq_error[-1]:0.3f} MHz")
            freqs.append(prog.data["new_freq"])
            params['ramsey_freq'] = err * 0.7 
            span = np.pi / np.abs(err * 0.7)
            span = np.min([span, auto_cfg.device.qubit.T2r[qi]*4])
            params['span'] = span
            config.update_qubit(cfg_dict["cfg_file"], "f_ge", prog.data["new_freq"], qi)
        elif prog.data["fit_err"] > max_err:
            
            if params['ramsey_freq'] == 'smart':
                print("Fit Error too high, increasing ramsey frequency")
                params['ramsey_freq'] = 0.2
            else:
                print("Fit Error too high, increasing ramsey frequency and span")
                params['ramsey_freq'] = 2*params['ramsey_freq']
                params['span']=auto_cfg.device.qubit.T2r[qi]*1.5
        else:
            print("Fit failed, trying spectroscopy.")
            qspec=find_spec(qi,cfg_dict,  start='fine')
            if style!='giveup':
                recenter(qi, cfg_dict, style='giveup')
            else:
                print('Failed to recenter')
            # if prog is not None and "new_ramsey_freq" in prog.data:
            #     params["ramsey_freq"] = prog.data["new_ramsey_freq"]
            #     params["span"] = np.pi / np.abs(prog.data["new_ramsey_freq"])
            #     params['soft_avgs']=2

        i += 1

    print(f"Change in frequency: {freqs[-1]-freqs[-1]:0.3f} MHz")
    print(freq_error)
    auto_cfg = config.load(cfg_dict["cfg_file"])
    end_freq = auto_cfg["device"]["qubit"]["f_ge"][qi]
    print(f"Qubit {qi} recentered from {start_freq} to {end_freq}")

    if i == ntries:
        return False
    else:
        return True
    
def meas_opt(cfg_dict, qubit_list, params=None, update=True, start_coarse=True):
    if params is None:
        params = {}
    cfg_path = cfg_dict['cfg_file']
    for qi in qubit_list: 
        do_more=True
        if start_coarse:
            while do_more:
                shotopt=meas.SingleShotOptExperiment(cfg_dict, qi=qi,params=params)
                do_more=shotopt.do_more
                if update:
                    config.update_readout(cfg_path, 'gain', shotopt.data['gain'], qi);
                    config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi);
        do_more=True
        while do_more:
            shotopt=meas.SingleShotOptExperiment(cfg_dict, qi=qi,params=params, style='fine')
            do_more=shotopt.do_more
            if update:
                config.update_readout(cfg_path, 'gain', shotopt.data['gain'], qi);
                config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi);
            