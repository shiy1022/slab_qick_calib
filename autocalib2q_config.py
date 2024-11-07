import experiments as meas
import config


def make_t1_cont(soc, expt_path, cfg_file, qubit_i, qubit_j, im=None, go=False, t1A=200, t1B=200, reps=1000000):
    
    t1_cont = meas.T1_2qbContinuous(
            soccfg=soc,
            path=expt_path,
            prefix=f"t1_continuous_2qubit{qubit_i, qubit_j}",
            config_file=cfg_file,
        )
    npts = 1

    t1_cont.cfg.expt = dict(
        startA= t1A / npts,  # wait time [us]
        startB = t1B / npts,  # wait time [us]
        step=0,
        expts=npts,
        reps=reps,  # number of times we repeat a time point
        rounds=1,  # number of start to finish sweeps to average over
        qubits=[qubit_i, qubit_j],
    )

    if go:
        t1_cont.go(analyze=False, display=False, progress=True, save=True)

    return t1_cont

def make_t1(soc, expt_path, cfg_file, qubit_i, qubit_j, im=None, go=False, span=600, npts=20, reps=200, rounds=1):
    t1 = meas.T1_2qbExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"t1_2qubit{qubit_i, qubit_j}",
        config_file=cfg_file,
        im=im
    )

    span = span
    npts = npts

    t1.cfg.expt = dict(
        startA=1,  # wait time [us]
        startB=1,  # wait time [us]
        start=1,
        step=span / npts,
        expts=npts,
        reps=reps,  # number of times we repeat a time point
        rounds=rounds,  # number of start to finish sweeps to average over
        qubits= [qubit_i, qubit_j],
        length_scan=span,  # length of the scan in us
        num_saved_points=1,  # number of points to save for the T1 continuous scan
    )

    if go:
        t1.go(analyze=False, display=False, progress=True, save=True)
    
    return t1