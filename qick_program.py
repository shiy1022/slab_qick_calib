from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D
from slab import AttrDict
import numpy as np
from qick import *


class QickProgram(AveragerProgramV2):
    def __init__(self, soccfg, final_delay=50, cfg={}):
        self.cfg = AttrDict(cfg)

        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, cfg=cfg)

    def _initialize(self, cfg, readout="standard"):
        cfg = AttrDict(self.cfg)

        self.qubits = cfg.expt.qubit
        q = self.qubits[0]

        self.adc_ch = cfg.hw.soc.adcs.readout.ch[q]
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[q]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q]
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[q]
        self.res_nqz = cfg.hw.soc.dacs.readout.nyquist[q]
        self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist[q]
        self.trig_offset = cfg.device.readout.trig_offset[q]
        

        if readout == "standard":
            self.readout_length = cfg.device.readout.readout_length[q]
            self.frequency = cfg.device.readout.frequency[q]
            self.gain = cfg.device.readout.gain[q]
            self.phase = cfg.device.readout.phase[q]
        if 'lo' in cfg.hw.soc and 'ch' in cfg.hw.soc.lo and cfg.hw.soc.lo.ch[q] != 'None':
            self.lo_ch = cfg.hw.soc.lo.ch[q]
            self.lo_nqz = cfg.hw.soc.lo.nyquist[q]
            self.mixer_freq=cfg.hw.soc.lo.mixer_freq[q]
            self.lo_gain = cfg.hw.soc.lo.gain[q]
            self.declare_gen(ch=self.lo_ch, nqz=self.lo_nqz, mixer_freq=self.mixer_freq-500)
            self.add_pulse(
                self.lo_ch,
                name="mix_pulse",
                style="const",
                length=self.readout_length,
                freq=self.mixer_freq,
                phase=0,
                gain=self.lo_gain,
            )
        else:
            self.lo_ch=None

        self.declare_gen(ch=self.res_ch, nqz=self.res_nqz)
        self.declare_readout(self.adc_ch, length=self.readout_length)
        self.add_readoutconfig(
            ch=self.adc_ch, name="readout", freq=self.frequency, gen_ch=self.res_ch
        )

        self.add_pulse(
            ch=self.res_ch,
            name="readout_pulse",
            style="const",
            ro_ch=self.adc_ch,
            length=self.readout_length,
            freq=self.frequency,
            phase=self.phase,
            gain=self.gain,
        )

        self.declare_gen(ch=self.qubit_ch, nqz=self.qubit_nqz)

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset, ddr4=True)

    def make_pulse(self, pulse, name):
        pulse = AttrDict(pulse)
        pulse_args = {
            "ch": self.qubit_ch,
            "name": name,
            "freq": pulse.freq,
            "phase": pulse.phase,
            "gain": pulse.gain,
        }

        if pulse.type == "gauss":
            style = "arb"
            self.add_gauss(
                ch=self.qubit_ch,
                name="ramp",
                sigma=pulse.sigma,
                length=pulse.sigma * pulse.sigma_inc,
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"
        else:
            style = "const"
            pulse_args["length"] = pulse.sigma
        pulse_args["style"] = style
        self.add_pulse(**pulse_args)

    def make_pi_pulse(self, q, freq, name):
        cfg = AttrDict(self.cfg)
        pulse = {
            key: value[q] for key, value in cfg.device.qubit.pulses[name].items()
        }
        pulse["freq"] = freq[q]
        pulse["phase"] = 0
        self.make_pulse(pulse, name)
        return pulse


def get_shots(self):

    for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
        offset = self.cfg.soc["readouts"][ch]["iq_offset"]
        nsamp = rocfg["length"]
        iq_raw = self.get_raw()
        i_shots = iq_raw[i][:, 0, 0] / nsamp - offset
        q_shots = iq_raw[i][:, 0, 1] / nsamp - offset
    return i_shots, q_shots
