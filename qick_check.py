"""
QICK Frequency Check Utilities

This module provides utility functions for checking and validating frequency settings
in quantum experiments using the QICK (Quantum Instrumentation Control Kit) framework.
It helps identify potential issues with frequency aliasing and validates qubit and
resonator frequency configurations.
"""

from . import config
import numpy as np


def check_freqs(i, cfg_dict):
    """
    Check qubit frequencies for potential aliasing issues and validate configuration.

    This function examines the qubit frequencies (f_ge and f_ef) for a specific qubit
    and checks for potential aliasing around the Nyquist frequency. It also calculates
    the anharmonicity and provides recommendations for frequency selection.

    Args:
        i (int): Qubit index to check
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Results are printed to the console
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get DAC channel for the specified qubit
    dac = auto_cfg.hw.soc.dacs.qubit.ch[i]

    # Get the sampling frequency from the SoC for the DAC channel
    fs = cfg_dict["soc"]._get_ch_cfg(dac)["fs"]

    # Calculate Nyquist frequency (mirror point)
    mirror_freq = fs / 2
    print(f"Nyquist frequency: {mirror_freq} MHz")

    # Get configured qubit frequencies
    freq = auto_cfg.device.qubit.f_ge[i]  # Ground to excited state transition
    fef = auto_cfg.device.qubit.f_ef[i]  # Excited to second excited state transition

    # Calculate potential aliased frequencies
    freq_offset = freq - mirror_freq
    alt_freq = mirror_freq - freq_offset  # Aliased g-e frequency

    freq_offset_ef = fef - mirror_freq
    alt_fef = mirror_freq - freq_offset_ef  # Aliased e-f frequency

    # Print frequency information
    print(f"Possible g-e frequencies: {freq} MHz and {alt_freq} MHz")
    print(f"Possible e-f frequencies: {fef} MHz and {alt_fef} MHz")

    # Calculate anharmonicity (difference between g-e and e-f transitions)
    alpha = freq - fef
    alpha2 = freq - alt_fef
    print(f"Anharmonicity: {alpha} MHz. With aliased e-f: {alpha2} MHz")

    # Provide recommendations based on frequency relationships
    if fef < freq and alt_fef < freq and fef > alt_freq and alt_fef > alt_freq:
        print(
            "Both e-f frequencies are less than the chosen g-e frequency and greater than the aliased g-e frequency."
        )
        print("Recommendation: Current g-e frequency is the correct choice.")

    if alt_fef > freq and alt_fef > alt_freq:
        print("Aliased e-f frequency is greater than both g-e frequencies.")
        print("Recommendation: Current e-f frequency is the correct choice.")


def check_resonances(cfg_dict):
    """
    Check resonator frequencies for potential aliasing issues.

    This function examines the readout resonator frequencies and identifies
    potential aliasing around the Nyquist frequency.

    Args:
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Results are printed to the console
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get readout DAC channel
    ro_dac = auto_cfg.hw.soc.dacs.readout.ch[0]

    # Get the sampling frequency from the SoC
    fs = cfg_dict["soc"]._get_ch_cfg(ro_dac)["fs"]

    # Calculate Nyquist frequency (mirror point)
    mirror_freq = fs / 2
    print(f"Nyquist frequency: {mirror_freq} MHz")

    # Get configured resonator frequencies
    freq = np.array(auto_cfg.device.readout.frequency)

    # Calculate aliased frequencies
    freq_offset = freq - mirror_freq
    alt_freq = mirror_freq - freq_offset

    # Print results
    print("Configured resonator frequencies (MHz):")
    print(freq)
    print("Aliased resonator frequencies (MHz):")
    print(alt_freq)


def check_adc(cfg_dict):
    """
    Check if any readout frequencies alias too close to Nyquist boundaries.

    This function identifies if any readout frequencies alias to within a critical
    window around Nyquist zone boundaries, which could cause readout issues.

    Args:
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Warnings are printed to the console if issues are found
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get ADC sampling frequency
    fs = cfg_dict["soc"]._get_ch_cfg(ro_ch=0)["fs"]
    print("ADC sampling frequency:", fs)

    # Calculate Nyquist frequency
    nyquist_freq = fs / 2

    # Get configured resonator frequencies
    freq = np.array(auto_cfg.device.readout.frequency)

    # Define critical window size (1/16 of sampling frequency for dynamic readout)
    window_size = fs / 16

    # Check each frequency for proximity to Nyquist zone boundaries
    for i, f in enumerate(freq):
        # Find which Nyquist zone the frequency falls in
        n = 0
        while abs(f - n * nyquist_freq) > nyquist_freq:
            n += 1

        # Calculate distance to nearest Nyquist zone boundary
        alias_dist = abs(f - n * nyquist_freq)

        # Warn if frequency is too close to a Nyquist zone boundary
        if alias_dist < window_size:
            print(
                f"Warning: Qubit {i} Frequency {f} MHz aliases to within {alias_dist:.1f} MHz of Nyquist frequency"
            )
            print(f"Distance to Nyquist zone {n} boundary: {alias_dist:.1f} MHz")
            print(f"This may cause readout issues. Consider adjusting the frequency.")
