import config 
import numpy as np 
# Fix me, check clock frequency using soc. 
def check_freqs(i, cfg_file):
    auto_cfg = config.load(cfg_file)
    mirror_freq = 9584.640/2
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

def check_resonances(cfg_file):
    auto_cfg = config.load(cfg_file)
    mirror_freq = 9584.640/2
    print('We mirror around ' + str(mirror_freq))
    freq = np.array(auto_cfg.device.readout.frequency)
    
    freq_offset = freq-mirror_freq

    alt_freq = mirror_freq - freq_offset
    print('List of mirrored resonator frequencies is:')
    print(alt_freq)