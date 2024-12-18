import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import AttrDict
from qick_experiment import QickExperiment

"""
Run this calibration when the wiring of the setup is changed.

This calibration measures the time of flight of measurement pulse so we only start capturing data from this point in time onwards. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
"""


class ToFCalibrationProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.soft_avgs = cfg.expt.reps  # same as reps
        self.cfg.reps = 1
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = self.cfg

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.dac_ch = cfg.hw.soc.dacs.readout.ch
        self.dac_ch_type = cfg.hw.soc.dacs.readout.type

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(
            self.frequency, gen_ch=self.dac_ch, ro_ch=self.adc_ch
        )  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.gain = cfg.expt.gain
        self.pulse_length = self.us2cycles(cfg.expt.pulse_length, gen_ch=self.dac_ch)
        self.readout_length = self.us2cycles(cfg.expt.readout_length, ro_ch=self.adc_ch)
        print(self.pulse_length, self.readout_length)

        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = self.adc_ch
        if self.dac_ch_type == "int4":
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.dac_ch_type == "mux4":
            assert self.dac_ch == 6
            mask = [0, 1, 2, 3]  # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0] * 4
            mux_freqs[cfg.expt.qubit] = cfg.expt.frequency
            mux_gains = [0] * 4
            mux_gains[cfg.expt.qubit] = cfg.expt.gain
        self.declare_gen(
            ch=self.dac_ch,
            nqz=cfg.hw.soc.dacs.readout.nyquist,
            mixer_freq=mixer_freq,
            mux_freqs=mux_freqs,
            mux_gains=mux_gains,
            ro_ch=ro_ch,
        )
        print(f"readout freq {mixer_freq} +/- {cfg.expt.frequency}")

        self.declare_readout(
            ch=self.adc_ch,
            length=self.readout_length,
            freq=self.frequency,
            gen_ch=self.dac_ch,
        )  # gen_ch links to the mixer_freq being used on the mux

        if self.dac_ch_type == "mux4":
            self.set_pulse_registers(
                ch=self.dac_ch, style="const", length=self.pulse_length, mask=mask
            )
        else:
            self.set_pulse_registers(
                ch=self.dac_ch,
                style="const",
                freq=self.freqreg,
                phase=0,
                gain=self.gain,
                length=self.pulse_length,
            )
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg = AttrDict(self.cfg)
        self.measure(
            pulse_ch=self.dac_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.expt.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.final_delay),
        )


# ====================================================== #


class ToFCalibrationExperiment(QickExperiment):
    """
    Time of flight experiment
    Experimental Config
    expt_cfg = dict(
        pulse_length [us]
        readout_length [us]
        gain [DAC units]
        frequency [MHz]
        adc_trig_offset [Clock ticks]
    }
    """

    def __init__(
        self,
        cfg_dict={},
        progress=None,
        prefix=None,
        qi=0,
        trig_offset=150,
        readout_length=1,
        reps=100,
        pulse_length=0.5,
        gain=None,
        go=True,
    ):
        if prefix is None:
            prefix = f"adc_trig_offset_calibration_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        self.cfg.expt = dict(
            pulse_length=pulse_length,  # [us]
            readout_length=readout_length,  # [us]
            trig_offset=0,  # [clock ticks]
            gain=self.cfg.device.readout.max_gain,
            frequency=self.cfg.device.readout.frequency[qi],  # [MHz]
            reps=reps,  # Number of averages per point
            qubit=qi,
            final_delay=0.1,
        )

        if go:
            self.go(analyze=False, display=False, progress=True, save=True)
            self.display(adc_trig_offset=trig_offset)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit
        super().update_config(q_ind=q_ind)

        data = {"i": [], "q": [], "amps": [], "phases": []}
        tof = ToFCalibrationProgram(soccfg=self.soccfg, cfg=self.cfg)
        iq = tof.acquire_decimated(
            self.im[self.cfg.aliases.soc], load_pulses=True, progress=True
        )
        i, q = iq[0]
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        data = dict(i=i, q=q, amps=amp, phases=phase)

        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        if data is None:
            data = self.data
        return data

    def display(self, data=None, adc_trig_offset=0, **kwargs):
        if data is None:
            data = self.data

        q_ind = self.cfg.expt.qubit
        adc_ch = self.cfg.hw.soc.adcs.readout.ch
        dac_ch = self.cfg.hw.soc.dacs.readout.ch
        plt.subplot(
            111,
            title=f"Time of flight calibration: DAC Ch. {dac_ch} to ADC Ch. {adc_ch}",
            xlabel="Clock ticks",
            ylabel="Transmission [ADC units]",
        )

        plt.plot(data["i"], label="I")
        plt.plot(data["q"], label="Q")
        plt.axvline(adc_trig_offset, c="k", ls="--")
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
