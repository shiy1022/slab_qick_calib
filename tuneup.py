import autocalib_config as cfg
import experiments as meas
import config
import matplotlib.pyplot as plt
import experiments.fitting as fitter
import numpy as np
import warnings

def check_chi(soc, expt_path, cfg_file, qubit_i, im=None, span=7, npts=251, plot=False, check_f=False):
    
    auto_cfg = config.load(cfg_file)
    freq = auto_cfg['device']['readout']['frequency'][qubit_i]
    start = freq-4.5
    center = start + span/2
    chi = cfg.make_rspec_fine(soc, expt_path, cfg_file, qubit_i, im=im, span=span, center =center, npts=npts, rounds=5, check_e = True, relax_delay=15, go=False)
    chi.go(analyze=True, display=False, progress=True, save=True)

    if check_f: 
        chif = cfg.make_rspec_fine(soc, expt_path, cfg_file, qubit_i, im=im, span=span, center = freq-0.65*span, npts=npts, rounds=5, check_e = True,check_f=True, relax_delay=15)
        chif.go(analyze=True, display=False, progress=True, save=True)
    rspec = cfg.make_rspec_fine(soc, expt_path, cfg_file, qubit_i, im=im, center = center, span=span, npts=npts, go=False)
    rspec.go(analyze=True, display=False, progress=True, save=True)

    if 'mixer_freq' in chi.cfg.hw.soc.dacs.readout:
            xpts_chi = chi.cfg.hw.soc.dacs.readout.mixer_freq + chi.data['xpts']
            xpts_res = rspec.cfg.hw.soc.dacs.readout.mixer_freq + chi.data['xpts']
            rspec.data['fit'][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
            chi.data['fit'][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
    else:
        xpts_chi = chi.data['xpts']
        xpts_res = rspec.data['xpts']

    arg=np.argmin(np.abs(np.abs(chi.data['amps']))-np.abs(rspec.data['amps']))
    arg2=np.argmin(np.abs(rspec.data['amps']))
    chi.data['rval']=rspec.data['xpts'][arg2]
    chi.data['cval']=chi.data['xpts'][arg]
    chi_val = xpts_chi[arg]-xpts_res[arg2]
    chi.data['freq_opt']=rspec.data['xpts'][arg]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].set_title(f'Chi Measurement Q{qubit_i}')
    ax[0].plot(xpts_res, rspec.data['amps'], label='No Pulse')
    ax[0].plot(xpts_chi, chi.data['amps'], label=f'e Pulse $\chi=${chi_val:0.2f} MHz')
    ax[0].legend()
    ax[0].axvline(x=xpts_chi[arg], color='r', linestyle='--')  # Add vertical line at selected point
    ax[0].axvline(x=xpts_res[arg2], color='r', linestyle='--')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('Frequency (MHz)')

    ax[1].plot(xpts_res, rspec.data['amps']-chi.data['amps'])
    ax[1].axvline(x=xpts_chi[arg], color='r', linestyle='--')
    ax[1].axvline(x=xpts_res[arg2], color='r', linestyle='--')
    ax[1].set_ylabel('Difference')
    ax[1].set_xlabel('Frequency (MHz)')

    ax[2].plot(xpts_res[1:-1], rspec.data['phase_fix'])
    ax[2].plot(xpts_chi[1:-1], chi.data['phase_fix'], label='e Pulse')

    

    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'][1:-1], chif.data['phase_fix'], label='f pulse')
    
    #ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *rspec.data["fit"]),'k')
    #ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *chi.data["fit"]),'k')
    # if check_f:
    #     ax[0].plot(chif.data['xpts'][1:-1], fitter.hangerS21func_sloped(chif.data["xpts"][1:-1], *chif.data["fit"]),'k')
    
    for a in ax:
        a.set_xlabel('Frequency (MHz)')
    
    
    ax[2].set_ylabel('Phase')
    plt.show()

    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'][1:-1], chif.data['phase_fix'], label='f pulse')

    #ax[0].plot(xpts_res, fitter.hangerS21func_sloped(xpts_res, *rspec.data["fit"]),'k')
    #ax[0].plot(xpts_chi, fitter.hangerS21func_sloped(xpts_chi, *chi.data["fit"]),'k')
    #if check_f:
    #    ax[0].plot(xpts_chi, fitter.hangerS21func_sloped(xpts_chi, *chif.data["fit"]),'k')
    
    

    imname = rspec.fname.split("\\")[-1]
    fig.savefig(rspec.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'chi.png')
    return [chi,rspec], chi_val, 

def measure_temp(soc, expt_path, cfg_path, qubit_i, im=None, npts=20, reps=None, rounds=None, chan=None):
    
    rabief=cfg.make_amprabiEF(soc, expt_path, cfg_path, qubit_i, im=im, go=True,reps=reps,rounds=rounds, pulse_ge=True)
    rabief_nopulse=cfg.make_amprabiEF(soc, expt_path, cfg_path, qubit_i, im=im, go=True,temp=True, pulse_ge=False, npts=npts, reps=reps, rounds=rounds)

    # To measure temperature, use fewer points to get more signal more quickly 
    h = 6.62607015e-34
    fge = 1e6*rabief.cfg.device.qubit.f_ge[qubit_i]
    kB = 1.380649e-23
    if chan is None:
        qubit_temp = 1e3*h*fge/(kB*np.log(rabief.data['best_fit'][0]/rabief_nopulse.data['best_fit'][0]))
        population = rabief_nopulse.data['best_fit'][0]/rabief.data['best_fit'][0]
    else:
        qubit_temp = 1e3*h*fge/(kB*np.log(rabief.data[chan][0]/rabief_nopulse.data[chan][0]))
        population = rabief_nopulse.data[chan][0]/rabief.data[chan][0]

    fig= plt.figure()
    i_best= str(rabief.data['i_best'])[2:-1]
    plt.plot(rabief.data['xpts'], rabief.data[i_best], label='With Pulse')
    plt.plot(rabief_nopulse.data['xpts'], rabief_nopulse.data[i_best], label='With Pulse')
    imname = rabief.fname.split("\\")[-1]
    fig.savefig(rabief.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'temp.png')
    print('Qubit temp [mK]:', qubit_temp)
    print('State preparation ratio:', population)

    print(rabief.data['best_fit'][0])
    print(rabief_nopulse.data['best_fit'][0])

    return qubit_temp, population

def recenter_smart(qi, cfg_dict, start='T2r', freq=0.3, max_err=0.45, min_r2=0.1, max_t1=500):

    # get original frequency 
    auto_cfg = config.load(cfg_dict['cfg_file'])
    start_freq = auto_cfg['device']['qubit']['f_ge'][qi]
    freqs=[start_freq]

    scans = [cfg.make_qspec, cfg.make_qspec, cfg.make_qspec, cfg.make_t2r, cfg.make_t2r] 
    params = [{'coarse':True, 'span':70}, {'span':25}, {'fine':True}, {'ramsey_freq':1.25*freq,'step':np.pi/np.abs(freq)/200, 'npts':200}, {'ramsey_freq':'smart', 'npts':200}]
    freq_error=[]
    scan_info = {'scans':scans, 'params':params}
    if start == 'T2r':
        level=3
    elif start == 'qspec':
        level=2
    else:
        level=0

    # Level 1: Qubit Spec coarse, set fge increase averaging 
    # Level 2: Qubit Spec medium, set fge increase averaging 
    # Level 3: Qubit Spec fine, set fge, kappa, run amp rabi 
    # Level 4: T2r coarse, set fge, T2r
    # Level 5: T2e coarse, set fge, T2e
    i=0
    all_done = False
    ntries=12
    while i<ntries and not all_done: 
        status, prog = run_scan_level(qi, cfg_dict, scan_info, level,min_r2=min_r2, max_t1=max_t1)
        if status:
            if level>2: 
                freq_error.append(prog.data['f_err'])
                print(f'New f error is {freq_error[-1]:0.3f} MHz')
            if level == len(scans)-1: 
                all_done = True
            else:
                level+=1
            
            freqs.append(prog.data['new_freq'])
        else:
            if prog is not None and 'new_ramsey_freq' in prog.data:
                params[3]['ramsey_freq']=prog.data['new_ramsey_freq']
                params[3]['step']=np.pi/np.abs(prog.data['new_ramsey_freq'])/200
                level=3
            else:
                level-=1
        ntries+=1
    
    print(f'Change in frequency: {freqs[-1]-freqs[-1]:0.3f} MHz')
    auto_cfg = config.load(cfg_dict['cfg_file'])
    end_freq = auto_cfg['device']['qubit']['f_ge'][qi]
    print(f'Qubit {qi} recentered from {start_freq} to {end_freq}')

    if i==ntries:
        return False
    else:
        return True
    
    
def find_spec(qi, cfg_dict, start='coarse', max_err=0.45, min_r2=0.1, max_t1=500):

    scans = [cfg.make_qspec, cfg.make_qspec, cfg.make_qspec] 
    params = [{'coarse':True, 'span':70}, {'span':25}, {'fine':True}]
    scan_info = {'scans':scans, 'params':params}
    if start == 'coarse':
        level=0
    else:
        level=2

    i=0
    all_done = False
    ntries=6
    while i<ntries and not all_done: 
        status, prog = run_spec(qi, cfg_dict, scan_info, level,min_r2=min_r2, max_t1=max_t1)
        if status:
            if level == len(scans)-1: 
                all_done = True
            else:
                level+=1   
                params[level]['center']=prog.data['new_freq']
        else:
            level-=1
        ntries+=1
    
    if i==ntries:
        return False
    else:
        return True
    
def run_spec(qi, cfg_dict, scan_info, level, min_r2=0.1, max_t1=500):

    status, prog = run_scan(scan_info['scans'][level], qi, cfg_dict, scan_info['params'][level], min_r2=min_r2)
    if status:
        config.update_qubit(cfg_dict['cfg_file'], 'f_spec', prog.data['new_freq'], qi)
    
    return status, prog    


def run_scan_level(qi, cfg_dict, scan_info, level, min_r2=0.1, max_t1=500):

    if level<=2:
        rounds=3
    else:
        rounds=1
    scan_info['params'][level]['rounds']=rounds
    status, prog = run_scan(scan_info['scans'][level], qi, cfg_dict, scan_info['params'][level], min_r2=min_r2)
    if level>2 and status:
        freq_arg = np.argmin(np.abs(prog.data['t2r_adjust']))
        freq = prog.data['t2r_adjust'][freq_arg]
        if np.abs(freq) > 2*np.abs(prog.cfg.expt.ramsey_freq):     
            status=False  
            prog.data['new_ramsey_freq']=1.3*freq
    if status:
        config.update_qubit(cfg_dict['cfg_file'], 'f_ge', prog.data['new_freq'], qi)
    
    # if qspec fine, run amp rabi 
    if level == 2 and status:
        status, amp_rabi = run_scan(cfg.make_amprabi,qi, cfg_dict, min_r2=min_r2)
        if status:
            config.update_qubit(cfg_dict['cfg_file'], ('pulses','pi_ge','gain'), int(amp_rabi.data['pi_length']), qi)            
        else:
            status=False
            prog=None
    
    return status, prog    
        
def recenter_qubit(qi,cfg_dict, freq=0.3, max_err=0.5, min_r2=0.1, max_t1=500, start='T2r'): 

    # get original frequency 
    auto_cfg = config.load(cfg_dict['cfg_file'])
    start_freq = auto_cfg['device']['qubit']['f_ge'][qi]

    if start == 'T2r':
        status, t2r = run_scan(cfg.make_t2r, qi, cfg_dict, {'ramsey_freq':1.5*freq,'step':np.pi/np.abs(freq)/200, 'npts':200}, min_r2, max_err)
    else:
        status=False 
    if status:
        # If first ramsey works, run a finer ramsey 
        config.update_qubit(cfg_dict['cfg_file'], 'f_ge', t2r.data['new_freq'], qi)
        config.update_qubit(cfg_dict['cfg_file'], 'T2r',t2r.data['best_fit'][3], qi, sig=2, rng_vals=[1.5, max_t1])
        status, t2r = run_scan(cfg.make_t2r, qi, cfg_dict, {'ramsey_freq':'smart'}, min_r2, max_err)
        freq_arg = np.argmin(np.abs(t2r.data['t2r_adjust']))
        freq = t2r.data['t2r_adjust'][freq_arg]
        if np.abs(freq) > 2*np.abs(t2r.cfg.expt.ramsey_freq):     
            status, t2r = recenter_qubit(qi,cfg_dict, freq=freq, min_r2=min_r2, max_t1=max_t1)   
        if status:
            config.update_qubit(cfg_dict['cfg_file'], 'f_ge', t2r.data['new_freq'], qi)
            config.update_qubit(cfg_dict['cfg_file'], 'T2r',t2r.data['best_fit'][3], qi, sig=2, rng_vals=[1.5, max_t1])
    else:
        status, qspec = run_scan(cfg.make_qspec, qi, cfg_dict, {'fine':True})
        if status:
            config.update_qubit(cfg_dict['cfg_file'], 'f_ge', qspec.data["best_fit"][2], qi)
            config.update_qubit(cfg_dict['cfg_file'], 'kappa', 2*qspec.data["best_fit"][3], qi)
            # then run amp rabi 
            status, amp_rabi = run_scan(cfg.make_amprabi,qi, cfg_dict, min_r2=min_r2)
            if status:
                config.update_qubit(cfg_dict['cfg_file'], ('pulses','pi_ge','gain'), int(amp_rabi.data['pi_length']), qi)            
            status, t2r=recenter_qubit(qi,cfg_dict,  freq=freq, max_err=max_err, min_r2=min_r2, max_t1=max_t1)
        else: 
            # Medium scale 
            status, qspec = run_scan(cfg.make_qspec, qi, cfg_dict, {'span':25})
            if status:
                config.update_qubit(cfg_dict['cfg_file'], 'f_ge', qspec.data["best_fit"][2], qi)
                status, t2r=recenter_qubit( qi,cfg_dict, freq=freq, max_err=max_err, min_r2=min_r2, max_t1=max_t1, start='qspec')
            else: 
                # Coarse scale 
                status, qspec = run_scan(cfg.make_qspec, qi, cfg_dict, {'coarse':True, 'span':70})
                if status:
                    config.update_qubit(cfg_dict['cfg_file'], 'f_ge', qspec.data["best_fit"][2], qi)
                    status, qspec = run_scan(cfg.make_qspec, qi, cfg_dict)
                    if status:
                        config.update_qubit(cfg_dict['cfg_file'], 'f_ge', qspec.data["best_fit"][2], qi)
                        status, t2r=recenter_qubit(qi, cfg_dict, freq=freq, max_err=max_err, min_r2=min_r2, max_t1=max_t1, start='qspec')
                    else:
                        status=False
                        t2r=None
                else:
                    print('Failed!')
                    status=False
                    t2r=None
        recenter_qubit( qi,cfg_dict, freq=freq, max_err=max_err, min_r2=min_r2, max_t1=max_t1)
    
    auto_cfg = config.load(cfg_dict['cfg_file'])
    end_freq = auto_cfg['device']['qubit']['f_ge'][qi]
    print(f'Qubit {qi} recentered from {start_freq} to {end_freq}')
    
    return status, t2r

    # If you are coming from Ramsey and had signal, try using the frequency you got to recenter iwth 
    # If not, use the ramsey with larger frequency 
    # First check ramsey with larger frequency - say 2 MHz 
    # If that converges, then try with smaller frequency 
    # If can't do ramsey, do qubit spec with 8 MHz span low pwoer 
    # If not, do it with medium power and 25 MHz 
    # If not do it with high power and 100 MHz
    # I think whenever it works, you then call it again. 

def run_scan(scan_name, qi, cfg_dict,params={}, min_r2=0.1, max_err=0.5):

    prog = scan_name(cfg_dict['soc'], cfg_dict['expt_path'], cfg_dict['cfg_file'], qi, cfg_dict['im'], **params)
    if 'fit_err' in prog.data and 'r2' in prog.data and prog.data['fit_err'] < max_err and prog.data['r2'] > min_r2:
        return True, prog
    elif 'fit_err' not in prog.data or 'r2' not in prog.data:
        return prog
    else:
        print('Fit failed')
        return False, prog 
        #suc = recenter_qubit(params[0:4])
        #if suc: 
        #    return run_scan(scan_name, params, min_r2, max_err)
        #else:
        #    warnings.warn('Fit failed and recentering did not succeed')
        #    return False, prog


    # Run the scan 
    # Check if values were good 
    # If not, run recentering 
    # If it succeeds, try to run scan 
    # If not, throw error or warning 
    # Return success status (perform updates outside? )

def get_coherence(scan_name, qi, cfg_dict, par, params={}, min_r2=0.1, max_err=0.5, tol=0.3, max_t1=500):
    # For t1, t2r, t2e
    err = 2*tol
    auto_cfg = config.load(cfg_dict['cfg_file'])
    old_par = auto_cfg['device']['qubit'][par][qi]
    i=0
    while err>tol and i<5: 
        status, prog = run_scan(scan_name, qi, cfg_dict, params, min_r2, max_err)
        if par == 'T1':
            new_par = prog.data['new_t1']
        else: 
            if 'best_fit' in prog.data:
                new_par = prog.data['best_fit'][3]
        if status:
            auto_cfg = config.update_qubit(cfg_dict['cfg_file'], par, new_par, qi, sig=2, rng_vals=[1.5, max_t1])
        elif not status:
            print('Failed')    
        err = np.abs(new_par-old_par)/old_par
        old_par = new_par
        i+=1

    return status, prog