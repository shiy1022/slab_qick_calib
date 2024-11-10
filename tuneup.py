import autocalib_config as cfg
import experiments as meas
import config
import matplotlib.pyplot as plt
import experiments.fitting as fitter
import numpy as np

def check_readout(soc, expt_path, cfg_file, qubit_i, im=None, span=2.5, npts=201, reps=650, check_f=False):
    chi = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, npts=npts, reps=reps)
    chi.go(analyze=True, display=False, progress=True, save=True)
    if check_f: 
        chif = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, npts=npts, reps=reps, check_f=True)
        chif.go(analyze=True, display=False, progress=True, save=True)
    rspec = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, reps=reps, npts=npts, check_e=False)
    rspec.go(analyze=True, display=False, progress=True, save=True)

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(rspec.data['xpts'], rspec.data['amps'], label='None')
    ax[2].plot(rspec.data['xpts'], rspec.data['avgi'])
    ax[3].plot(rspec.data['xpts'], rspec.data['avgq'])

    ax[0].plot(chi.data['xpts'], chi.data['amps'], label='e pulse')
    ax[2].plot(chi.data['xpts'], chi.data['avgi'], label='e pulse')
    ax[3].plot(chi.data['xpts'], chi.data['avgq'], label='e pulse')

    ax[1].plot(rspec.data['xpts'], rspec.data['amps']-chi.data['amps'])
    arg=np.argmax(np.abs(rspec.data['amps']-chi.data['amps']))
    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'], chif.data['avgi'], label='f pulse')
        ax[3].plot(chif.data['xpts'], chif.data['avgq'], label='f pulse')

    ax[0].plot(rspec.data['xpts'][1:-1], fitter.hangerS21func_sloped(rspec.data["xpts"][1:-1], *rspec.data["fit"]),'k')
    ax[0].plot(chi.data['xpts'][1:-1], fitter.hangerS21func_sloped(chi.data["xpts"][1:-1], *chi.data["fit"]),'k')
    ax[0].plot(chif.data['xpts'][1:-1], fitter.hangerS21func_sloped(chif.data["xpts"][1:-1], *chif.data["fit"]),'k')
    
    ax[0].legend()

    ax[0].axvline(x=rspec.data['xpts'][arg], color='r', linestyle='--')  # Add vertical line at selected point
    ax[1].axvline(x=rspec.data['xpts'][arg], color='r', linestyle='--')
    plt.show()

    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'], chif.data['avgi'], label='f pulse')
        ax[3].plot(chif.data['xpts'], chif.data['avgq'], label='f pulse')

    ax[0].plot(rspec.data['xpts'][1:-1], fitter.hangerS21func_sloped(rspec.data["xpts"][1:-1], *rspec.data["fit"]),'k')
    ax[0].plot(chi.data['xpts'][1:-1], fitter.hangerS21func_sloped(chi.data["xpts"][1:-1], *chi.data["fit"]),'k')
    ax[0].plot(chif.data['xpts'][1:-1], fitter.hangerS21func_sloped(chif.data["xpts"][1:-1], *chif.data["fit"]),'k')
    
    ax[0].legend()
    chi.data['chi'] = (rspec.data['fit'][0]-chi.data['fit'][0])/2
    chi.data['freq_opt']=rspec.data['xpts'][arg]
    return chi 

def measure_temp(soc, expt_path, cfg_path, qubit_i, im=None, npts=20, reps=650, rounds=100, chan='fit_avgi'):
    
    rabief=cfg.make_amprabiEF(soc, expt_path, cfg_path, qubit_i, im=im, go=True, pulse_ge=True)
    rabief_nopulse=cfg.make_amprabiEF(soc, expt_path, cfg_path, qubit_i, im=im, go=True, pulse_ge=False, reps=reps, rounds=rounds, npts=npts)

    # To measure temperature, use fewer points to get more signal more quickly 
    h = 6.62607015e-34
    fge = 1e6*rabief.cfg.device.qubit.f_ge[qubit_i]
    kB = 1.380649e-23
    qubit_temp = 1e3*h*fge/(kB*np.log(rabief.data[chan][0]/rabief_nopulse.data[chan][0]))
    population = rabief_nopulse.data[chan][0]/rabief.data[chan][0]
    print('Qubit temp [mK]:', qubit_temp)
    print('State preparation ratio:', population)

    print(rabief.data['fit_avgq'][0])
    print(rabief_nopulse.data['fit_avgq'][0])

    print(rabief.data['fit_amps'][0])
    print(rabief_nopulse.data['fit_amps'][0])
    return qubit_temp, population