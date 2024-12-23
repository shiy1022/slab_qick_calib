import autocalib_config as cfg
import experiments as meas
import config
import matplotlib.pyplot as plt
import fitting as fitter
import numpy as np
import scipy.constants as cs
import warnings
from datetime import datetime

update_readout=False
max_t1=500
max_err = 1
min_r2 = 0.35
tol=0.3
auto_cfg = config.load(cfg_path)
plt.rcParams['legend.handlelength'] = 1.0

def tune_up_qubit(qi, cfg_dict, update=True, first_time=False, readout=True, single=False, temp=False, max_t1=500, max_err=1, recenter=True, min_r2=0.35, tol=0.3, im=False):
        cfg_path = cfg_dict['cfg_path']

        rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa'})
        if update_readout and rspec.status: 
            auto_cfg = config.update_readout(cfg_path, 'frequency', rspec.data['fit'][0], qi)
        if update and rspec.status:
            auto_cfg = config.update_readout(cfg_path, 'qi', rspec.data['fit'][1], qi)
            auto_cfg = config.update_readout(cfg_path, 'qe', rspec.data['fit'][2], qi)
            auto_cfg = config.update_readout(cfg_path, 'kappa', rspec.data['kappa'], qi, rng_vals=[0.03, 10])

        qspec=meas.QubitSpec(cfg_dict, qi=qi, style='fine', params={'span':3,'npts':85,'soft_avgs':3, 'len':'t1'})        
        if np.abs(qspec.data['new_freq']-auto_cfg.device.qubit.f_ge[qi])>0.25 and qspec.status:
            print('Qubit frequency is off spectroscopy by more than 250 kHz, recentering')
            auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data['new_freq'], qi)
        
        # Amp Rabi to get pi pulse; recenter
        amp_rabi = meas.AmplitudeRabiExperiment(cfg_dict,qi=qi)
        if update:
            if amp_rabi.status:
                config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), int(amp_rabi.data['pi_gain']), qi)
        if not amp_rabi.status:
            print(f'Amplitude Rabi fit failed on Q{qi}')
            if recenter:
                status = recenter_smart(qi, cfg_dict, start='qspec')
                if status:
                    amp_rabi = meas.AmplitudeRabiExperiment(cfg_dict,qi=qi)
                    if update and amp_rabi.status:
                        config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), int(amp_rabi.data['pi_length']), qi)
                    else:
                        print(f'Amplitude Rabi fit failed on Q{qi}')
        
        # Run T1 to get sense of coherence times
        meas.T1Experiment(cfg_dict, qi=qi)
        if update and t1.status: 
            auto_cfg = config.update_qubit(cfg_path, 'T1', t1.data['new_t1_i'], qi,sig=2, rng_vals=[1.5, max_t1*2])
            if first_time:
                auto_cfg = config.update_qubit(cfg_path, 'T2r', t1.data['new_t1_i'], qi,sig=2, rng_vals=[1.5, max_t1*2])
                auto_cfg = config.update_readout(cfg_path, 'final_delay', 6*t1.data['new_t1'], qi, sig=2,rng_vals=[10, 1000])
                auto_cfg = config.update_qubit(cfg_path, 'T2e', 2*t1.data['new_t1'], qi,sig=2, rng_vals=[1.5, max_t1*2])

        # Run single shot opt to improve readout 

        # Run Single shot 

        # Run Ramsey first to center

        # Amp rabi to improve pi pulse

        # Run SS Opt fine and SS 

        # Run Ramsey for coherence 

        # Run T1 

        # Run T2 Echo 

        # Run chi 

        if first_time:
            status = tuneup.recenter_smart(qi,cfg_dict, freq=0.3, min_r2=min_r2, max_t1=max_t1)
            
            status, amp_rabi = tuneup.run_scan(cfg.make_amprabi,qi, cfg_dict, min_r2=min_r2)
            if update and status:
                config.update_qubit(cfg_path, ('pulses','pi_ge','gain'), int(amp_rabi.data['pi_length']), qi)
            elif not status:
                print('Amplitude Rabi fit failed')
                status = tuneup.recenter_smart(qi, cfg_dict)
                
        if single:
            # if first_time:
            #     shotopt=tuneup.run_scan(cfg.make_singleshot_opt,qi,cfg_dict, {'npts_f':1, 'npts_gain':7, 'npts_len':7})
            #     config.update_readout(cfg_path, 'gain', int(shotopt.data['gain']), qi)
            #     config.update_readout(cfg_path, 'readout_length', shotopt.data['length'], qi);

            shotopt=tuneup.run_scan(cfg.make_singleshot_opt,qi,cfg_dict, {'npts_f':1, 'npts_gain':5, 'npts_len':5, 'fine':True})
            if update:
                config.update_readout(cfg_path, 'gain', int(shotopt.data['gain']), qi);
                config.update_readout(cfg_path, 'readout_length', float(shotopt.data['length']), qi);

        shot = tuneup.run_scan(cfg.make_singleshot,qi, cfg_dict)
        if update:
            config.update_readout(cfg_path, 'phase', shot.data['angle'], qi,sig=3);
            config.update_readout(cfg_path, 'threshold', shot.data['thresholds'][0], qi, sig=3);
            config.update_readout(cfg_path, 'fidelity', shot.data['fids'][0], qi);

        # Get T2 ramsey 
        status, t2r = tuneup.run_scan(cfg.make_t2r,qi, cfg_dict, min_r2=min_r2,params={'ramsey_freq':'smart'})
        freq_arg = np.argmin(np.abs(t2r.data['t2r_adjust']))
        freq = t2r.data['t2r_adjust'][freq_arg]
        if update and recenter: 
            if np.abs(freq) > 2*np.abs(t2r.cfg.expt.ramsey_freq):     
                status = tuneup.recenter_smart(qi,cfg_dict, freq=freq, min_r2=min_r2, max_t1=max_t1)            
            elif status:
                config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], qi)
            elif t2r.data['r2']>min_r2:
                status, t2r = tuneup.run_scan(cfg.make_t2r,qi, cfg_dict, min_r2=min_r2,params={'ramsey_freq':0.2})
                if status: 
                    auto_cfg = config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], qi)
            else:
                print('T2 Ramsey fit failed')
            status, t2r = tuneup.get_coherence(cfg.make_t2r, qi, cfg_dict, 'T2r', {'ramsey_freq':'smart'}, min_r2=min_r2, tol=tol, max_t1=max_t1)
            if first_time:
                auto_cfg = config.update_qubit(cfg_path, 'T1', t2r.data['best_fit'][3], qi, rng_vals=[1.5, max_t1])

        # Get T1 and T2 echo
        if update: 
            status, t1 = tuneup.get_coherence(cfg.make_t1, qi, cfg_dict,'T1', min_r2=min_r2,tol=tol, max_t1=max_t1)
            if first_time:
                auto_cfg = config.update_readout(cfg_path, 'final_delay', 6*t1.data['new_t1'], qi, sig=2,rng_vals=[10, 1000])
                auto_cfg = config.update_qubit(cfg_path, 'T2e', 2*t1.data['new_t1'], qi,sig=2, rng_vals=[1.5, max_t1*2])
            status, t2e = tuneup.get_coherence(cfg.make_t2e, qi, cfg_dict,'T2e', {'ramsey_freq':'smart'}, min_r2=min_r2, tol=tol, max_t1=max_t1)
        else:
            tuneup.run_scan(cfg.make_t1,qi, cfg_dict)
            tuneup.run_scan(cfg.make_t2e,qi, cfg_dict, {'ramsey_freq':'smart'})

        # Get chi
        chid, chi_val=tuneup.check_chi(soc, expt_path, cfg_path, qi, im=im)
        auto_cfg = config.update_readout(cfg_path, 'chi', float(chi_val), qi)


def make_summary_figure(cfg_dict, progs, qi):    
    fig, ax = plt.subplots(3,3, figsize=(14,9))
    amp_rabi.display(ax=[ax[0,0]], savefig=False)
    t1.display(ax=[ax[0,1]], savefig=False)
    t2r.display(ax=[ax[1,0]], savefig=False)
    t2e.display(ax=[ax[1,1]], savefig=False)
    shot.display(ax=[ax[0,2],ax[1,2]], plot=False)
    rspec.display(ax=[ax[2,0]])

    cap = f'Length: {auto_cfg.device.readout.readout_length[qi]:0.2f} $\mu$s'
    cap += f'\nGain: {auto_cfg.device.readout.gain[qi]}'
    ax[0,2].text(0.02, 0.05, cap, transform=ax[0,2].transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))       
    
    qspec.display(ax=[ax[2,2]], plot_all=False)
    ax[2,2].set_title(f'Qubit Freq: {auto_cfg.device.qubit.f_ge[qi]:0.2f} MHz')
    ax[2,2].axvline(x=auto_cfg.device.qubit.f_ge[qi], color='k', linestyle='--')


    ax[2,1].set_title(f'Chi Measurement Q{qi}')
    ax[2,1].plot(chid[1].data['xpts'], chid[1].data['amps'], label='No Pulse')
    ax[2,1].plot(chid[0].data['xpts'], chid[0].data['amps'], label=f'e Pulse')
    cap=f'$\chi=${chi_val:0.2f} MHz'
    ax[2,1].text(0.04, 0.35, cap, transform=ax[2,1].transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    ax[2,1].axvline(x=chid[0].data['cval'], color='k', linestyle='--')  # Add vertical line at selected point
    ax[2,1].axvline(x=chid[0].data['rval'], color='k', linestyle='--')
    ax[2,1].legend()
    ax[2,1].set_ylabel('Amplitude')
    ax[2,1].set_xlabel('Frequency (MHz)')

    fig.tight_layout()
    datestr = datetime.now().strftime("%Y%m%d_%H%M")
    fname = cfg_dict['expt_path'] + f'images\\summary\\qubit{qi}_tuneup_{datestr}.png'
    print(fname)
    fig.savefig(fname)
    plt.show()
    %matplotlib inline


def find_spec(qi, cfg_dict, start="coarse", max_err=0.45, min_r2=0.1):

    style = ["huge","coarse", "medium", "fine"]
    level = style.index(start)

    i = 0
    all_done = False
    ntries = 6
    while i < ntries and not all_done:        
        print(level)
        prog = meas.QubitSpec(cfg_dict, qi=qi, min_r2=min_r2, params={}, style=style)
        if prog.status:
            config.update_qubit(cfg_dict["cfg_file"], "f_ge", prog.data["new_freq"], qi)
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
            auto_cfg.device.readout.spec_gain[qi] = auto_cfg.device.readout.spec_gain[qi]*2 
            auto_cfg.device.readout.reps[qi] = auto_cfg.device.readout.spec_reps[qi]*2
            level=0
        i += 1

    if i == ntries:
        return False
    else:
        return True
    

def get_coherence(
    scan_name,
    qi,
    cfg_dict,
    par,
    params={},
    min_r2=0.1,
    max_err=0.5,
    tol=0.3,
    max_t1=500,
    ):
    # For t1, t2r, t2e
    err = 2 * tol
    auto_cfg = config.load(cfg_dict["cfg_file"])
    old_par = auto_cfg["device"]["qubit"][par][qi]
    i = 0
    while err > tol and i < 5:
        prog = scan_name(cfg_dict, qi, params=params)
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
            params["span"] = 2 * new_par # Usually occurs because too little signal 
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
