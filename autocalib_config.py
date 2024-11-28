import experiments as meas
import config
import matplotlib.pyplot as plt
import experiments.fitting as fitter
import numpy as np
max_gain = 32768
reps_base=150 
rounds_base=1
reps_base_spec=150
rounds_base_spec=1

def safe_gain(gain):
    gain = np.min([gain, max_gain])
    return gain

def make_tof(soc, expt_path, cfg_file, qubit_i, im=None, go=True):

    tof = meas.ToFCalibrationExperiment(soccfg=soc,
    path=expt_path,
    prefix=f"adc_trig_offset_calibration_qubit{qubit_i}",
    config_file=cfg_file, 
    im=im)

    tof.cfg.expt = dict(pulse_length=0.5, # [us]
    readout_length=1.0, # [us]
    trig_offset=0, # [clock ticks]
    gain=max_gain,
    frequency=tof.cfg.device.readout.frequency[qubit_i], # [MHz]
    reps=1000, # Number of averages per point
    qubit=qubit_i) 

    tof.cfg.device.readout.relax_delay[qubit_i]=0.1 # wait time between experiments [us]

    if go: 
        tof.go(analyze=False, display=False, progress=True, save=True)
        tof.display(adc_trig_offset=160) 
    
    return tof

def make_rspec_coarse(soc, expt_path, cfg_file, qubit_i, im=None, start=7000, span=250, reps=800, npts=5000, gain=0.2, rounds=1):
    rspec = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_coarse",
    config_file=cfg_file,   
    im=im
    )

    rspec.cfg.expt = dict(
        start = start, #Lowest resonator frequency
        step=span/npts, # min step ~1 Hz
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
        gain=gain,
        rounds=rounds,
        qubit_chan=rspec.cfg.hw.soc.adcs.readout.ch[qubit_i]
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]
    rspec.go(analyze=False, display=False, progress=True, save=True)

    rspec.analyze(fit=False, findpeaks = True)
    rspec.display(fit=False, findpeaks = True)
    return rspec

def make_rspec_fine(soc, expt_path, cfg_file, qubit_i, im=None, go=True, center=None, span=5, npts=200, reps=None, rounds=None, gain=None, smart=False):
    
    prog = meas.ResonatorSpectroscopyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"resonator_spectroscopy_res{qubit_i}",
    config_file=cfg_file,  
    im=im, 
    )
    if gain is None: 
        gain = prog.cfg.device.readout.gain[qubit_i]
    if center==None: 
        center = prog.cfg.device.readout.frequency[qubit_i]
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec)
    if span=='kappa':
        span = float(prog.cfg.device.readout.kappa[qubit_i]*5)

    prog.cfg.expt = dict(
        start = center-span/2, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        smart=smart,
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
        gain=gain,
        rounds=rounds
    )

    prog.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]

    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_rspec_2d(cfg_dict, qubit_i=0, go=True, center=None, span=5, npts=200, reps=500, gain=0.05, smart=False, pts=1000):
    rspec = meas.Resonator2DSpectroscopyExperiment(
    soccfg=cfg_dict['soc'],
    path=cfg_dict['expt_path'],
    prefix=f"resonator_2dspectroscopy_res{qubit_i}",
    config_file=cfg_dict['cfg_file'],  
    im=cfg_dict['im'], 
    )

    if center==None: 
        center = rspec.cfg.device.readout.frequency[qubit_i]

    rspec.cfg.expt = dict(
        start = center-span/2, #Lowest resontaor frequency
        step=span/npts, # min step ~1 Hz
        smart=smart,
        pts=pts,
        expts=npts, # Number experiments stepping from start
        reps= reps, # Number averages per point 
        pulse_e=False, # add ge pi pulse prior to measurement
        pulse_f=False, # add ef pi pulse prior to measurement
        qubit=qubit_i,
        qubit_chan=rspec.cfg.hw.soc.adcs.readout.ch[qubit_i],
        gain=gain
    )

    rspec.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]

    if go: 
        rspec.go(analyze=False, display=True, progress=True, save=True)

    return rspec

def make_rpowspec(soc, expt_path, cfg_file, qubit_i, res_freq, im=None, span_f=15, npts_f=200, span_gain=27000, start_gain=5000, npts_gain=10, reps=None, rounds=None, smart=False, log=True, rng=200):

    prog = meas.ResonatorPowerSweepSpectroscopyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ResonatorPowerSweepSpectroscopyExperiment_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    
    if reps is None:
        reps = prog.cfg.device.readout.reps[qubit_i]*reps_base_spec/10000
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec)


    prog.cfg.expt = dict(
        start_f = res_freq-span_f/2, # resonator frequency to be mixed up [MHz]
        step_f = span_f/npts_f, # min step ~1 Hz, 
        smart = smart, 
        expts_f=npts_f, # Number experiments stepping freq from start
        start_gain=start_gain,
        step_gain=span_gain/npts_gain, # Gain step size
        expts_gain=npts_gain+1, # Number experiments stepping gain from start
        reps= reps, # Number averages per point
        pulse_e=False, # add ge pi pulse before measurement
        pulse_f=False, # add ef pi pulse before measurement
        qubit=qubit_i,  
        log=log,
        rng=rng,
        rounds=rounds,
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    ) 
    

    prog.cfg.device.readout.relax_delay = 5 # Wait time between experiments [us]    
    prog.cfg.device.readout.readout_length = 5
    return prog

def make_chi(soc, expt_path, cfg_file, qubit_i, im=None, go=False, span=3, npts=251, reps=None, rounds=None, check_e=True, check_f=False, smart=False):
    # This adds an e pulse first 

    if check_f: 
        prefix = f"resonator_spectroscopy_chi_qubit{qubit_i}_f"
    else:
        prefix = f"resonator_spectroscopy_chi_qubit{qubit_i}"
    
    prog = meas.ResonatorSpectroscopyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=prefix,
        config_file=cfg_file,
        im=im
        )
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec/2)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec)
    prog.cfg.expt = dict(
        start=prog.cfg.device.readout.frequency[qubit_i]-span/2, # MHz
        # start=rspec_chi.cfg.device.readout.frequency[qubit_i]-rspec_chi.cfg.device.readout.lo_sideband[qubit_i]*span, # MHz
        step=span/npts,
        expts=npts,
        reps=reps,
        gain = prog.cfg.device.readout.gain[qubit_i],
        pulse_e=check_e, # add ge pi pulse prior to measurement
        pulse_f=check_f, # add ef pi pulse prior to measurement
        qubit=qubit_i,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
        smart=smart, 
        rounds=rounds
    )
    # rspec_chi.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)
    
    return prog

def make_qspec(soc, expt_path, cfg_file, qubit_i, im=None, span=None, npts=None, reps=None, rounds=None, gain=None, coarse=False, ef=False, len=50):    
# This one may need a bunch of options. 
# coarse: wide span, medium gain, centered at ge freq
# ef: coarse: medium span, extra high gain, centered at the ef frequency  
# otherwise, narrow span, low gain, centered at ge frequency 

    if coarse and span is None:
        span=800 
        prefix = f"qubit_spectroscopy_coarse_qubit{qubit_i}"
        if npts is None: 
            npts = 500
    elif span is None:
        span=3
        prefix = f"qubit_spectroscopy_fine_qubit{qubit_i}"
        if npts is None:
            npts = 200
    else:
        prefix = f"qubit_spectroscopy_qubit{qubit_i}"
        if npts is None:
            npts = 500

    if coarse is True and gain is None:
        gain=10000
    elif gain is None:
        gain=100
    

    prog = meas.PulseProbeSpectroscopyExperiment(
    soccfg=soc,
    path = expt_path, 
    prefix = prefix,
    config_file=cfg_file,
    im=im
    )
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec)
    if ef:
        freq = prog.cfg.device.qubit.f_ef[qubit_i]
        if coarse:
            prefix = f"qubit_spectroscopy_qubit_coarse_ef{qubit_i}"
            span=450
        else:
            prefix = f"qubit_spectroscopy_qubit_fine_ef{qubit_i}"
    else:
        freq = prog.cfg.device.qubit.f_ge[qubit_i]

    
    prog.cfg.expt = dict(
        start= freq-span/2, # qubit frequency to be mixed up [MHz]
        step = span/npts, # min step ~1 Hz
        expts = npts, # Number experiments stepping from start
        reps = reps, # Number averages per point
        rounds = rounds, #Number of start to finish sweeps to average over 
        length = len, # qubit probe constant pulse length [us]
        gain = gain, #qubit pulse gain  
        pulse_type = 'const', 
        #pulse_type = 'gauss',  
        qubit = qubit_i,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    ) 

    prog.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
    return prog

def make_qspec_power(soc, expt_path, cfg_file, qubit_i, im=None, span=None, npts=500, reps=None, rounds=None, wide=True, expts_gain=7, rng=100, len=50):
    prefix = f"qubit_spectroscopy_power_qubit{qubit_i}"
    prog = meas.PulseProbePowerSweepSpectroscopyExperiment(
    soccfg=soc,
    path = expt_path, 
    prefix = prefix,
    config_file=cfg_file,
    im=im
    )
    
    if wide: 
        freq = prog.cfg.device.qubit.f_ge[qubit_i]-150
        if span is None:
            span=800
    else:
        freq = prog.cfg.device.qubit.f_ge[qubit_i]
        if span is None:
            span=20

    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec/2)
        rounds = np.max([rounds,1])

    prog.cfg.expt = dict(
        start_f= freq-span/2, # qubit frequency to be mixed up [MHz]
        step_f = span/npts, # min step ~1 Hz
        expts_f = npts, # Number experiments stepping from start
        reps = reps, # Number averages per point
        rounds = rounds, #Number of start to finish sweeps to average over 
        gain_pts=expts_gain,
        length = len, # qubit probe constant pulse length [us]
        pulse_type = 'const', 
        #pulse_type = 'gauss',  
        expts_gain=expts_gain,
        qubit = qubit_i,
        log=True,
        rng=rng,
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    ) 

    prog.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
    return prog

def make_qspec_ef(soc, expt_path, cfg_file, qubit_i, im=None, go=False, span=None, npts=None, reps=None, rounds=None, gain=None, coarse=False):

    if coarse and span is None:
        span=500 
        prefix = f"qubit_spectroscopy_coarse_ef_qubit{qubit_i}"
    elif span is None:
        span=5
        prefix = f"qubit_spectroscopy_fine_ef_qubit{qubit_i}"
    else:
        prefix = f"qubit_spectroscopy_qubit_ef_{qubit_i}"

    if coarse is True and gain is None:
        gain=20000
    elif gain is None:
        gain=200
    
    if coarse is True and npts is None: 
        npts = 500
    elif npts is None:
        npts = 100


    prog = meas.PulseProbeEFSpectroscopyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=prefix,
        config_file=cfg_file,
        im=im
    )
    if reps is None:
        if coarse is True:
            reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec)
        else:
            reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base_spec*3)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base_spec)

    prog.cfg.expt = dict(
        start=prog.cfg.device.qubit.f_ef[qubit_i]-0.5*span, # resonator frequency to be mixed up [MHz]
        step=span/npts, # min step ~1 Hz
        expts=npts, # Number of experiments stepping from start
        reps=reps, # Number of averages per point
        rounds=rounds, # Number of start to finish sweeps to average over
        length=1, # ef probe constant pulse length [us]
        gain=gain, # ef pulse gain
        pulse_type='gauss', # ef pulse type
        qubit=qubit_i,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    # qEFspec.cfg.device.readout.relax_delay = 500 # Wait time between experiments [us]
    
    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_lengthrabi(soc, expt_path, cfg_file, qubit_i, im=None, npts = 100, reps = None, gain = 2000, num_pulses = 1, step=None, rounds=None):
    prog = meas.LengthRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"length_rabi_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    if step is None:
        soc.cycles2us(1)
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    prog.cfg.expt = dict(
        start =  0.0025, 
        step= step, # [us] this is the samllest possible step size (size of clock cycle)
        expts= npts, 
        reps= reps,
        gain =  gain, #qubit gain [DAC units]
        #gain=lengthrabi.cfg.device.qubit.pulses.pi_ge.gain[qubit_i],
        pulse_type='gauss',
        # pulse_type='const',
        checkZZ=False,
        checkEF=False, 
        qubits=[qubit_i],
        num_pulses = 1, #number of pulses to play, must be an odd number
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
        rounds = rounds
    )

    return prog

def make_amprabi(soc, expt_path, cfg_file, qubit_i, im=None, go=False, sigma=None, npts = 100, reps = None, rounds=None, gain=None, checkZZ=False):
    #auto_cfg.device.qubit.pulses.pi_ge.gain[qubit_i]
    prog = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"amp_rabi_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
        )

    if sigma is None: 
        sigma=prog.cfg.device.qubit.pulses.pi_ge.sigma[qubit_i] # gaussian sigma for pulse length - overrides config [us]
    if gain is None:
        gain = prog.cfg.device.qubit.pulses.pi_ge.gain[qubit_i]*4
        gain = int(safe_gain(gain))
    span = gain
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    prog.cfg.expt = dict(     
        start=0,
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        sigma_test= sigma,
        checkZZ=checkZZ,
        checkEF=False, 
        checkCC=False,
        qubits=[qubit_i],
        pulse_type='gauss',
        # pulse_type='const',
        num_pulses = 1, #number of pulses to play, must be an odd number in order to achieve a pi rotation at pi length/ num_pulses,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)
    
    return prog

def make_amprabi_zz(soc, expt_path, cfg_file, qubits, im=None, go=False, sigma=None, npts = 100, reps = None, rounds=None, gain=None, checkZZ=False):
    #auto_cfg.device.qubit.pulses.pi_ge.gain[qubit_i]
    qubit_i=qubits[0]
    qubit_j=qubits[1]
    prog = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"amp_rabi_ZZ_qubit{qubit_i}{qubit_j}",
        config_file=cfg_file,
        im=im
        )

    if sigma is None: 
        sigma=prog.cfg.device.qubit.pulses.pi_ge.sigma[qubit_i] # gaussian sigma for pulse length - overrides config [us]
    if gain is None:
        gain = prog.cfg.device.qubit.pulses.pi_ge.gain[qubit_i]*4
        gain = int(safe_gain(gain))
    span = gain
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    prog.cfg.expt = dict(     
        start=0,
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        sigma_test= sigma,
        checkZZ=checkZZ,
        checkEF=False, 
        checkCC=False,
        qubits=qubits,
        pulse_type='gauss',
        # pulse_type='const',
        num_pulses = 1, #number of pulses to play, must be an odd number in order to achieve a pi rotation at pi length/ num_pulses,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)
    
    return prog

def make_amprabi_cc(soc, expt_path, cfg_file, qubits, im=None, go=False, sigma=None, npts = 100, reps = None, rounds=None, gain=None):
    #auto_cfg.device.qubit.pulses.pi_ge.gain[qubit_i]
    prog = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"amp_rabi_qubit{qubits[0]}_drive{qubits[1]}read",
        config_file=cfg_file,
        im=im
        )
    qubit_i=qubits[0]
    qubit_j=qubits[1]
    if sigma is None: 
        sigma=prog.cfg.device.qubit.pulses.pi_ge.sigma[qubit_i] # gaussian sigma for pulse length - overrides config [us]
    if gain is None:
        gain = prog.cfg.device.qubit.pulses.pi_ge.gain[qubit_i]*4
        gain = int(safe_gain(gain))
    span = gain
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_j]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_j]*rounds_base)
    prog.cfg.expt = dict(     
        start=0,
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        sigma_test= sigma,
        checkZZ=False,
        checkEF=False, 
        checkCC=True,
        qubits=qubits,
        pulse_type='gauss',
        # pulse_type='const',
        num_pulses = 1, #number of pulses to play, must be an odd number in order to achieve a pi rotation at pi length/ num_pulses,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_j],
    )

    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)
    
    return prog

def make_amprabi_chevron(soc, expt_path, cfg_file, qubit_i, im=None, span_gain=30000, npts_gain=75, start_gain=1000, span_f=20, npts_f=40, reps=None, rounds=None, sigma=0.2, go=True):
    prog = meas.AmplitudeRabiChevronExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"amp_rabi_qubit_chevron{qubit_i}",
        config_file=cfg_file,
        im=im
    )
    
    if reps is None:
        reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)

    prog.cfg.expt = dict(
        start_f=prog.cfg.device.qubit.f_ge[qubit_i]-span_f/2,
        step_f=span_f/(npts_f-1),
        expts_f=npts_f,
        start_gain=start_gain,
        step_gain=int(span_gain/npts_gain), # [dac level]
        expts_gain=npts_gain,
        reps=reps,
        rounds=rounds,
        sigma_test=sigma, # gaussian sigma for pulse length - overrides config [us]
        checkZZ=False,
        checkEF=False, 
        pulse_ge=False,
        qubits=[qubit_i],
        pulse_type='gauss',
        num_pulses=1,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
        # pulse_type='adiabatic',
        # mu=6, # dimensionless
        # beta=4, # dimensionless
        # sigma_test=0.120*4, # us
    )

    # amprabichev.cfg.device.readout.relax_delay = 50 # Wait time between experiments [us]
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)
    return prog

def make_t2r(soc, expt_path, cfg_file, qubit_i, im=None, go=False, npts = 100, reps = None, rounds=None, step=None, ramsey_freq=0.1, checkEF=False):
    prog = meas.RamseyExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ramsey_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    #ramsey_freq=npts/t2_1/8, npts=npts, reps=250, step=t2_1/npts
    if step is None: 
        span = 2*prog.cfg.device.qubit.T2r[qubit_i]
        step = span/npts
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(2*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)

    if ramsey_freq=='smart':
        ramsey_freq = np.pi/2/prog.cfg.device.qubit.T2r[qubit_i]

    prog.cfg.expt = dict(
        start=0, # wait time tau [us]
        #step=soc.cycles2us(10), # [us] make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds, 
        qubits=[qubit_i],
        qubit=qubit_i,
        checkZZ=False,
        checkEF=checkEF,
        acStark=False,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_t2r_stark(soc, expt_path, cfg_file, qubit_i, im=None, go=False, npts = 100, reps = None, rounds=None, step=None, ramsey_freq=0.1, gain=None, freq=None, df=20):
    prog = meas.RamseyStarkExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ramsey_stark_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    #ramsey_freq=npts/t2_1/8, npts=npts, reps=250, step=t2_1/npts
    if step is None: 
        span = 2*prog.cfg.device.qubit.T2r[qubit_i]
        step = span/npts
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(2*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    if gain is None: 
        gain=100
    if freq is None: 
        freq = prog.cfg.device.qubit.f_ge[qubit_i]-df
    if ramsey_freq=='smart':
        ramsey_freq = np.pi/2/prog.cfg.device.qubit.T2r[qubit_i]

    prog.cfg.expt = dict(
        start=soc.cycles2us(1000), # wait time tau [us]
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds, 
        qubits=[qubit_i],
        qubit=qubit_i,
        stark_gain=gain, 
        stark_freq=freq,
        checkZZ=False,
        checkEF=False,
        acStark=True,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_t2r_stark_freq(soc, expt_path, cfg_file, qubit_i, im=None, go=False, npts = 100, reps = None, rounds=None, step=None, ramsey_freq=0.1, gain=None,  span_f=30, npts_f=10):
    prog = meas.RamseyStarkFreqExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ramsey_stark_freq_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    #ramsey_freq=npts/t2_1/8, npts=npts, reps=250, step=t2_1/npts
    if step is None: 
        span = 2*prog.cfg.device.qubit.T2r[qubit_i]
        step = span/npts
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(2*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    if gain is None: 
        gain=100
    if ramsey_freq=='smart':
        ramsey_freq = np.pi/2/prog.cfg.device.qubit.T2r[qubit_i]

    prog.cfg.expt = dict(
        start=soc.cycles2us(1000), # wait time tau [us]
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds,
        start_f=prog.cfg.device.qubit.f_ge[qubit_i]+1,
        step_f=span_f/(npts_f-1),
        expts_f=npts_f, 
        qubits=[qubit_i],
        qubit=qubit_i,
        stark_gain=gain, 
        checkZZ=False,
        checkEF=False,
        acStark=True,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog


def make_t2r_stark_amp(soc, expt_path, cfg_file, qubit_i, im=None, go=False, npts = 100, reps = None, rounds=None, step=None, ramsey_freq=0.1, df=20,  span_gain=10000, npts_gain=10, start_gain=0):
    prog = meas.RamseyStarkPowerExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ramsey_stark_freq_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    #ramsey_freq=npts/t2_1/8, npts=npts, reps=250, step=t2_1/npts
    if step is None: 
        span = 2*prog.cfg.device.qubit.T2r[qubit_i]
        step = span/npts
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(2*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    
    if ramsey_freq=='smart':
        ramsey_freq = np.pi/2/prog.cfg.device.qubit.T2r[qubit_i]

    prog.cfg.expt = dict(
        start=soc.cycles2us(1000), # wait time tau [us]
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds,
        stark_freq = prog.cfg.device.qubit.f_ge[qubit_i]-df,
        start_gain=start_gain,
        step_gain=span_gain/(npts_gain-1),
        expts_gain=npts_gain, 
        qubits=[qubit_i],
        qubit=qubit_i,
        checkZZ=False,
        checkEF=False,
        acStark=True,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_t2e(soc, expt_path, cfg_file, qubit_i, im=None, go=False, npts = 100, reps = None, rounds=None, ramsey_freq=0.05, step=None):

    prog = meas.RamseyEchoExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"echo_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
        )
    
    if step is None: 
        span = 2*prog.cfg.device.qubit.T2e[qubit_i]
        step = span/npts
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(2*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    prog.cfg.expt = dict(
        start=0.1, #soc.cycles2us(150), # total wait time b/w the two pi/2 pulses [us]
        step=step, #step,
        expts=npts,
        ramsey_freq=ramsey_freq, # frequency by which to advance phase [MHz]
        num_pi=1, # number of pi pulses
        cpmg=False, # set either cp or cpmg to True
        cp=True, # set either cp or cpmg to True
        reps=reps,
        rounds=rounds,
        qubit=qubit_i,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go: 
        prog.go(analyze=True, display=True, progress=True, save=True)
    return prog

def make_t1(soc, expt_path, cfg_file, qubit_i, im=None, go=False, span=None, npts=60, reps=None, rounds=None, fine=False):

    span = span 
    npts = npts
    
    prog = meas.T1Experiment(
      soccfg=soc,
      path=expt_path,
      prefix=f"t1_qubit{qubit_i}",
      config_file= cfg_file,
      im=im
    )
    if span is None: 
        span = 3*prog.cfg.device.qubit.T1[qubit_i]
    if reps is None:
        reps = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)

    if fine is True: 
        rounds = rounds*2
    prog.cfg.expt = dict(
        start=0, # wait time [us]
        step=span/npts, 
        expts=npts,
        reps=reps, # number of times we repeat a time point 
        rounds=rounds, # number of start to finish sweeps to average over
        qubit=qubit_i,
        length_scan = span, # length of the scan in us
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_t1_2d(soc, expt_path, cfg_file, qubit_i, im=None, go=False, span=600, npts=200, reps=None, rounds=1, sweep_pts=100):

    span = span 
    npts = npts
    
    prog = meas.T1_2D(
      soccfg=soc,
      path=expt_path,
      prefix=f"t1_2d_qubit{qubit_i}",
      config_file= cfg_file,
      im=im
    )
    if reps is None:
        reps = int(3*prog.cfg.device.readout.reps[qubit_i]*reps_base)

    prog.cfg.expt = dict(
        start=0, # wait time [us]
        step=int(span/npts), 
        expts=npts,
        reps=reps, # number of times we repeat a time point 
        rounds=rounds, # number of start to finish sweeps to average over
        qubit=qubit_i,
        length_scan = span, # length of the scan in us
        sweep_pts = sweep_pts, # number of points to sweep over,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog

def make_t1doub(soc, expt_path, cfg_file, qubit_i, im=None, go=False, delay_time=150, npts=1, reps=1000, rounds=1):

    t1 = meas.T1ContinuousDoub(
      soccfg=soc,
      path=expt_path,
      prefix=f"t1cont2_qubit{qubit_i}",
      config_file= cfg_file,
      im=im
    )

    t1.cfg.expt = dict(
        start=0, # wait time [us]
        step=delay_time/npts, 
        expts=npts,
        reps=reps, # number of times we repeat a time point 
        rounds=rounds, # number of start to finish sweeps to average over
        qubit=qubit_i,
        length_scan = delay_time, # length of the scan in us
        num_saved_points = 10, # number of points to save for the T1 continuous scan 
    )

    if go:
        t1.go(analyze=False, display=False, progress=True, save=True)


    return t1    

def make_t1_cont(soc, expt_path, cfg_file, qubit_i, reps=2000000, norm=False, delay_time=150, rounds=1):
    if norm:
        prefix = f"t1_continuous_qubit_norm{qubit_i}"
    else:
        f"t1_continuous_qubit{qubit_i}"
    t1_cont = meas.T1Continuous(
        soccfg=soc,
        path=expt_path,
        prefix=prefix,
        config_file=cfg_file,
    )

    npts = 1
    if norm: 
        t1_cont.cfg.expt = dict(
            start=0,  # wait time [us]
            step=delay_time,
            expts=2,
            reps=reps,  # number of times we repeat a time point
            rounds=rounds,  # number of start to finish sweeps to average over
            qubit=qubit_i,
            )
           
    else:
        t1_cont.cfg.expt = dict(
        start=delay_time,  # wait time [us]
        step=0,
        expts=npts,
        reps= reps,  # number of times we repeat a time point
        rounds=1,  # number of start to finish sweeps to average over
        qubit=qubit_i,
        )
        


    return t1_cont

def make_singleshot(soc, expt_path, cfg_file, qubit_i, im=None, go=False, reps=10000, check_f=False):

    prog = meas.HistogramExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"single_shot_qubit{qubit_i}",
    config_file= cfg_file,
    im=im,
    )

    prog.cfg.expt = dict(
        reps=reps,
        check_e = True, 
        check_f=check_f,
        qubits=[qubit_i],
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)


    return prog

def make_singleshot_opt(soc, expt_path, cfg_file, qubit_i, go=True, im=None, reps=10000, start_f = None, span_f=0.5, npts_f=5, start_gain=None, span_gain=None, npts_gain=5, start_len=None, span_len=None, npts_len=5, check_f=False, fine=False):

    prog = meas.SingleShotOptExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"single_shot_opt_qubit{qubit_i}",
        config_file=cfg_file, 
        im=im
    )

    if npts_f==1 and start_f is None:
        start_f = prog.cfg.device.readout.frequency[qubit_i]
    elif start_f is None:
        start_f = prog.cfg.device.readout.frequency[qubit_i] - 0.5*span_f

    if npts_gain==1 and start_gain is None:
        start_gain = prog.cfg.device.readout.gain[qubit_i]
    elif start_gain is None:
        if fine: 
            start_gain = prog.cfg.device.readout.gain[qubit_i] * 0.8
        else:    
            start_gain = prog.cfg.device.readout.gain[qubit_i] * 0.3

    if npts_len==1 and start_len is None:
        start_len = prog.cfg.device.readout.readout_length[qubit_i]
    elif start_len is None:
        if fine: 
            start_len = prog.cfg.device.readout.readout_length[qubit_i]*0.8
        else:       
            start_len = prog.cfg.device.readout.readout_length[qubit_i]*0.3

    if npts_f == 1:
        step_f =0 
    else:
        step_f = span_f/(npts_f-1)

    if span_gain is None: 
        if fine: 
            span_gain = 0.4 * prog.cfg.device.readout.gain[qubit_i]
        else:
            span_gain = 1.8 * prog.cfg.device.readout.gain[qubit_i]

    if span_gain + start_gain > max_gain:
        span_gain = max_gain - start_gain

    if npts_gain == 1:
        step_gain = 0
    else:
        step_gain = span_gain/(npts_gain-1)

    if span_len is None:
        if fine: 
            span_len = 0.4 * prog.cfg.device.readout.readout_length[qubit_i]
        else:
            span_len = 1.8 * prog.cfg.device.readout.readout_length[qubit_i]

    if npts_len == 1:
        step_len =0 
    else:
        step_len = span_len/(npts_len-1)


    prog.cfg.expt = dict(
        reps=reps,
        qubit=qubit_i,
        start_f=start_f,
        step_f=step_f,
        expts_f=npts_f,
        start_gain=start_gain,#start_gain=1000,
        step_gain=step_gain,
        expts_gain=npts_gain,
        start_len=start_len,
        step_len=step_len,
        expts_len=npts_len,
        check_f=check_f,
        save_data=True,
        qubits=[qubit_i],
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )

    if go:
        prog.go(analyze=False, display=False, progress=False, save=True)
        prog.analyze()
        prog.display()

    return prog

def make_amprabiEF(soc, expt_path, cfg_file, qubit_i, im=None, go=False, span=30000, npts=101, reps=None, rounds=None, pulse_ge=True, temp=False):
    if pulse_ge:
        prefix = "amp_rabi_EF_ge" +f"_qubit{qubit_i}"
    else:
        prefix ="amp_rabi_EF"+f"_qubit{qubit_i}"

    prog = meas.AmplitudeRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=prefix,
        config_file=cfg_file,        
        im=im
    )

    if reps is None:
        if temp: 
            reps = int(10*prog.cfg.device.readout.reps[qubit_i]*reps_base)
        else:
            reps = int(prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        if temp: 
            rounds = int(100*prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
        else:
            rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)

    prog.cfg.expt = dict(
        start=0, # qubit gain [dac level]
        step=int(span/npts), # [dac level]
        expts=npts,
        reps=reps,
        rounds=rounds,
        pulse_type='gauss',
        qubits=[qubit_i],
        # sigma_test=0.013, # gaussian sigma for pulse length - default from cfg [us]
        checkZZ=False,
        checkEF=True, 
        num_pulses=1,
        pulse_ge=pulse_ge, 
        qubit_chan=prog.cfg.hw.soc.adcs.readout.ch[qubit_i])

    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)
    return prog

def make_acstark(soc, expt_path, cfg_file, qubit_i, span_f=100, npts_f=300, span_gain=10000, npts_gain=25, im=None, go=True):
    acspec = meas.ACStarkSelfShiftPulseProbeExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"ac_stark_shift_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    acspec.cfg.expt = dict(        
        start_f=acspec.cfg.device.qubit.f_ge[qubit_i]-0.25*span_f, # Pulse frequency [MHz]
        step_f=span_f/npts_f,
        expts_f=npts_f,
        start_gain=0, 
        step_gain=int(span_gain/npts_gain),
        expts_gain=npts_gain+1,
        pump_freq=acspec.cfg.device.readout.frequency[qubit_i]+50,
        # pump_freq=acspec.cfg.device.qubit.f_EgGf[2],
        pump_length=10, # [us]
        qubit_length=1, # [us]
        qubit_gain=500,
        pulse_type='const',
        reps=100,
        rounds=10, # Number averages per point
        qubit=qubit_i,
        qubit_chan=acspec.cfg.hw.soc.adcs.readout.ch[qubit_i]
    )
    acspec.cfg.device.readout.relax_delay = 25
    return acspec

def make_rb(soc, expt_path, cfg_file, qubit_i, im=None, go=False):
    

    rb = meas.SingleRB(
      soccfg=soc,
      path=expt_path,
      prefix=f"rb_qubit{qubit_i}",
      config_file= cfg_file,
    )

    rb.cfg.expt = dict(
        qubit= qubit_i,
        singleshot_reps= 10000,   # single shot measurement repetitions
        span= 50,   # single shot plot span
        reps= 100,
        rounds= 10,
        variations= 20,   # number of different sequences
        rb_depth= 5,    # rb sequence depth
        IRB_gate_no= -1   # IRB gate number, -1 means not using
    )

    if go:
        rb.go(analyze=True, display=True, progress=True, save=True)

    return rb