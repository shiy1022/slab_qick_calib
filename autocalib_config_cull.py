
def make_amprabi_cc(soc, expt_path, cfg_file, qubits, im=None, go=True, sigma=None, npts = 100, reps = None, rounds=None, gain=None):
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

def make_t2r_stark_freq(soc, expt_path, cfg_file, qubit_i, im=None, go=True, npts = 100, reps = None, rounds=None, step=None, ramsey_freq=0.1, gain=None,  span_f=30, npts_f=10, start=0.1):
    prog = meas.RamseyStarkFreq2Experiment(
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
        start=start, # wait time tau [us]
        step= step, # [us]
        expts=npts,
        ramsey_freq=ramsey_freq, # [MHz]
        reps=reps,
        rounds=rounds,
        start_f=prog.cfg.device.qubit.f_ge[qubit_i]+5,
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
   

def make_t1_stark_amp_cont(soc, expt_path, cfg_file, qubit_i, im=None, go=True, npts = 200, reps = None, rounds=None,  freq=None, df=40, acStark=True, stop_gain=32768, start_gain=0, delay_time=None):
    prog = meas.T1StarkPowerContExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"t1_stark_cont_qubit{qubit_i}",
        config_file=cfg_file,
        im=im
    )

    if delay_time is None: 
        delay_time = prog.cfg.device.qubit.T1[qubit_i]
    if reps is None:
        reps = int(10*prog.cfg.device.readout.reps[qubit_i]*reps_base)
        repsE = int(2*prog.cfg.device.readout.reps[qubit_i]*reps_base)
    if rounds is None:
        rounds = int(prog.cfg.device.readout.rounds[qubit_i]*rounds_base)
    if freq is None: 
        freq = prog.cfg.device.qubit.f_ge[qubit_i]+df

    prog.cfg.expt = dict(
        expts=npts,
        repsT1=reps,
        repsE=repsE,
        rounds=rounds, 
        qubit=qubit_i,
        start_gain=start_gain,
        stop_gain=stop_gain,
        stark_freq=freq,
        acStark=acStark, 
        checkZZ=False,
        checkEF=False,
        delay_time=delay_time,
        qubit_chan = prog.cfg.hw.soc.adcs.readout.ch[qubit_i],
    )
    if go:
        prog.go(analyze=True, display=True, progress=True, save=True)

    return prog
 
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