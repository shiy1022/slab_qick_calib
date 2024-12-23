import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
from slab import AttrDict
from qick_experiment import QickExperiment
from qick_program import QickProgram

"""
Run this calibration when the wiring of the setup is changed.

This calibration measures the time of flight of measurement pulse so we only start capturing data from this point in time onwards. Time of flight (tof) is stored in parameter cfg.device.readout.trig_offset.
"""


class LoopbackProgram(QickProgram):
    
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay, cfg)
    
    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.readout_length = cfg.expt.readout_length
        self.phase = 0
        super()._initialize(cfg, readout="custom")
        print('check')

    def _body(self, cfg):
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=0,
            ddr4=True,
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
        params={},
        go=True,
    ):
        if prefix is None:
            prefix = f"adc_trig_offset_calibration_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
        params_def = {
            "soft_avgs": 1000,
            "pulse_length": 0.5,  # [us]
            "readout_length": 1,  # [us]
            "trig_offset": self.cfg.device.readout.trig_offset[qi],  # [us]
            "gain": self.cfg.device.readout.max_gain,
            "frequency": self.cfg.device.readout.frequency[qi],  # [MHz]
            "reps": 1,  # Number of averages per point
            "qubit": [qi],
            "final_delay": 0.1,
        }
        self.cfg.expt = {**params_def, **params}

        if go:
            self.go(analyze=False, display=False, progress=True, save=True)
            self.display(adc_trig_offset=self.cfg.expt.trig_offset)

    def acquire(self, progress=False):
        
        final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        prog = LoopbackProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
        iq_list = prog.acquire_decimated(self.im[self.cfg.aliases.soc],
            soft_avgs=self.cfg.expt.soft_avgs,)
        t = prog.get_time_axis(ro_index=0)
        i  = iq_list[0][:,0]
        q  = iq_list[0][:,1]
        plt.show()
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        data = {'xpts':t, 'i':i, 'q':q, 'amps':amp, 'phases':phase}

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

        q_ind = self.cfg.expt.qubit[0]
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        dac_ch = self.cfg.hw.soc.dacs.readout.ch[q_ind]
        plt.subplot(
            111,
            title=f"Time of flight calibration: DAC Ch. {dac_ch} to ADC Ch. {adc_ch}",
            xlabel="Clock ticks",
            ylabel="Transmission [ADC units]",
        )

        plt.plot(data["xpts"], data["i"], label="I")
        plt.plot(data["xpts"], data["q"], label="Q")
        plt.axvline(adc_trig_offset, c="k", ls="--")
        plt.legend()
        plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
