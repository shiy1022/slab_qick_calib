import autocalib_config as cfg
import experiments as meas
import config
import matplotlib.pyplot as plt
import experiments.fitting as fitter
import numpy as np

def check_readout(soc, expt_path, cfg_file, qubit_i, im=None, span=8, npts=201, check_f=False):
    chi = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, npts=npts)
    chi.go(analyze=True, display=False, progress=True, save=True)
    if check_f: 
        chif = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, npts=npts, check_f=True)
        chif.go(analyze=True, display=False, progress=True, save=True)
    rspec = cfg.make_chi(soc, expt_path, cfg_file, qubit_i, im=im, span=span, npts=npts, check_e=False)
    rspec.go(analyze=True, display=False, progress=True, save=True)
    if 'mixer_freq' in chi.cfg.hw.soc.dacs.readout:
            xpts = chi.cfg.hw.soc.dacs.readout.mixer_freq + chi.data['xpts']
            rspec.data['fit'][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
            chi.data['fit'][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
    else:
        xpts = chi.data['xpts']
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(xpts, rspec.data['amps'], label='No Pulse')
    ax[2].plot(xpts[1:-1], rspec.data['phase_fix'])

    ax[0].set_title(f'Chi Measurement Q{qubit_i}')

    ax[0].plot(xpts, chi.data['amps'], label='e pulse')
    ax[2].plot(xpts[1:-1], chi.data['phase_fix'], label='e pulse')

    ax[1].plot(xpts, rspec.data['amps']-chi.data['amps'])
    arg=np.argmax(np.abs(rspec.data['amps'])-np.abs(chi.data['amps']))
    arg2=np.argmin(np.abs(rspec.data['amps'])-np.abs(chi.data['amps']))
    chi_val = xpts[arg2]-xpts[arg]
    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'][1:-1], chif.data['phase_fix'], label='f pulse')
    
    #ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *rspec.data["fit"]),'k')
    #ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *chi.data["fit"]),'k')
    # if check_f:
    #     ax[0].plot(chif.data['xpts'][1:-1], fitter.hangerS21func_sloped(chif.data["xpts"][1:-1], *chif.data["fit"]),'k')
    
    ax[0].legend()

    ax[0].axvline(x=xpts[arg], color='r', linestyle='--')  # Add vertical line at selected point
    ax[1].axvline(x=xpts[arg], color='r', linestyle='--')
    ax[1].axvline(x=xpts[arg2], color='r', linestyle='--')
    ax[0].axvline(x=xpts[arg2], color='r', linestyle='--')

    plt.show()

    if check_f:
        ax[0].plot(chif.data['xpts'], chif.data['amps'], label='f pulse')
        ax[2].plot(chif.data['xpts'][1:-1], chif.data['phase_fix'], label='f pulse')

    ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *rspec.data["fit"]),'k')
    ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *chi.data["fit"]),'k')
    if check_f:
        ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *chif.data["fit"]),'k')
    
    ax[0].legend()
    chi.data['chi'] = (rspec.data['fit'][0]-chi.data['fit'][0])/2
    chi.data['freq_opt']=rspec.data['xpts'][arg]


    imname = rspec.fname.split("\\")[-1]
    fig.savefig(rspec.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'chi.png')
    return chi, chi_val

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