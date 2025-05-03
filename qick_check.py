import slab_qick_calib.config as config 
import numpy as np 
# Fix me, check clock frequency using soc. 
def check_freqs(i, cfg_dict):
    auto_cfg = config.load(cfg_dict['cfg_file'])
    dac = auto_cfg.hw.soc.dacs.qubit.ch[i]
    # Get the correct sampling frequency from soc for the DAC channel
    fs = cfg_dict['soc']._get_ch_cfg(dac)['fs']
    mirror_freq = fs/2
    print('We mirror around ' + str(mirror_freq))
    freq = auto_cfg.device.qubit.f_ge[i]
    fef = auto_cfg.device.qubit.f_ef[i]

    freq_offset = freq-mirror_freq

    alt_freq = mirror_freq - freq_offset

    freq_offset_ef = fef-mirror_freq

    alt_fef = mirror_freq - freq_offset_ef

    print('Possible frequencies are ' +str(freq) + ' and ' + str(alt_freq))
    print('Possible ef frequencies are ' +str(fef) + ' and ' + str(alt_fef))
    alpha = freq-fef 
    alpha2 = freq - alt_fef
    print('Anharmonicity is ' + str(alpha) + '. With alt_ef it is ' + str(alpha2))
    if fef < freq and alt_fef < freq and fef > alt_freq and alt_fef > alt_freq: 
        print('Both ef frequencies are less than chosen freq and greater than alt freq, so freq is correct choice')
    if alt_fef > freq and alt_fef > alt_freq: 
        print('Alt ef is greater than freq and alt freq, so ef is correct choice')

def check_resonances(cfg_dict):
    auto_cfg = config.load(cfg_dict['cfg_file'])
    # Get readout DAC channel 
    ro_dac = auto_cfg.hw.soc.dacs.readout.ch[0]
    # Get the correct sampling frequency from soc
    fs = cfg_dict['soc']._get_ch_cfg(ro_dac)['fs']
    mirror_freq = fs/2
    print('We mirror around ' + str(mirror_freq))
    freq = np.array(auto_cfg.device.readout.frequency)
    
    freq_offset = freq-mirror_freq

    alt_freq = mirror_freq - freq_offset
    print('List of mirrored resonator frequencies is:')
    print(alt_freq)

def check_adc(cfg_dict):
    auto_cfg = config.load(cfg_dict['cfg_file'])
    fs= cfg_dict['soc']._get_ch_cfg(ro_ch=0)['fs']
    nyquist_freq = fs/2
    freq = np.array(auto_cfg.device.readout.frequency)
    window_size = fs/16 # This is true for dynamic readout 
    #print(window_size)
    # Check if any frequency aliases fall within window around nyquist frequency
    for i, f in enumerate(freq):
        n = 0
        while abs(f - n*nyquist_freq) > nyquist_freq:
            n += 1
        
        alias_dist = abs(f - n*nyquist_freq)
        #print(alias_dist)
        if alias_dist < window_size:
            print(f"Warning: Qubit {i} Frequency {f} MHz aliases to within {alias_dist:.1f} MHz of Nyquist frequency")
            print(f"Distance to Nyquist zone {n} boundary: {alias_dist:.1f} MHz")


    


