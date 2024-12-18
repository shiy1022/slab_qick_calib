from qick.asm_v2 import AveragerProgramV2
from qick.asm_v2 import QickSpan, QickSweep1D
from slab import AttrDict
import numpy as np
from qick import *


class QickResProgram(AveragerProgramV2):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        self.cfg.soft_avgs = cfg.expt.soft_avgs
        super().__init__(soccfg, self.cfg)

    def _initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)

        self.qubits = self.cfg.expt.qubit
        q = self.qubits[0]

        self.adc_ch = cfg.hw.soc.adcs.readout.ch[q]
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[q]

        self.frequency = cfg.expt.frequency[q]
        self.gain = cfg.expt.gain[q]
        self.readout_length = cfg.device.readout.readout_length[q]
        self.phase = cfg.device.readout.phase[q]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q])
        self.declare_readout(self.adc_ch, length=self.readout_length)

        self.add_pulse(
            ch=self.res_ch,
            name="readout_pulse",
            ro_ch=self.adc_ch,
            length=self.readout_length,
            freq=self.frequency,
            phase=self.phase,
            gain=self.gain,
        )

        self.add_loop("freq_loop", cfg.expt.steps)

    def _body(self):
        cfg = AttrDict(self.cfg)
        self.send_readout_config(ch=self.adc_ch, name="my_readout", t=0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[cfg.adc_ch], pins=[0], t=cfg.device.readout.trig_offset[q])


class QickPulseProgram(AveragerProgramV2):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    def _initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)

        self.qubits = self.cfg.expt.qubit
        q = self.qubits[0]

        self.adc_ch = cfg.hw.soc.adcs.readout.ch[q]
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[q]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q]
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[q]

        # Readout
        self.frequency = cfg.expt.frequency[q]
        self.gain = cfg.expt.gain[q]
        self.readout_length = cfg.device.readout.readout_length[q]
        self.phase = cfg.device.readout.phase[q]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q])
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

        # Qubit
        self.f_ge = cfg.device.qubit.f_ge[q]
        self.pi_sigma = cfg.device.qubit.pulses.pi_ge.sigma[q]
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain[q]
        self.qubit_phase = 0
        if self.cfg.expt.pulse_f:
            self.f_ef = cfg.device.qubit.f_ef[q]
            self.pi_ef_sigma = cfg.device.qubit.pulses.pi_ef.sigma[q]
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain[q]

        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q])
        self.add_pulse(
            ch=qubit_ch,
            name="qubit_pulse",
            ro_ch=ro_ch,
            style="const",
            length=self.pi_sigma,
            freq=self.f_ge,
            phase=self.qubit_phase,
            gain=self.pi_gain,
        )

    def _body(self):
        cfg = AttrDict(self.cfg)
        self.send_readout_config(ch=self.adc_ch, name="readout", t=0)
        self.pulse(ch=cfg.qubit_ch, name="qubit_pulse", t=0)
        self.delay_auto(t=0.02, tag="waiting")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[cfg.adc_ch], pins=[0], t=cfg.device.readout.trig_offset[q])


class QickProgram(AveragerProgramV2):
    def __init__(self, soccfg, final_delay, cfg):
        self.cfg = AttrDict(cfg)

        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        self.qubits = cfg.expt.qubit
        q = self.qubits[0]

        self.adc_ch = cfg.hw.soc.adcs.readout.ch[q]
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[q]
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q]
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[q]

        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist[q])
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

        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist[q])
        pulse_args = {
            "ch": self.qubit_ch,
            "name": "qubit_pulse",
            "freq": self.qubit_freq,
            "phase": self.qubit_phase,
            "gain": self.qubit_gain,
        }

        if self.pulse_type == "gauss":
            style = "arb"
            self.add_gauss(
                ch=self.qubit_ch,
                name="ramp",
                sigma=self.qubit_ramp,
                length=self.qubit_length,
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"
        else:
            style = "const"
            pulse_args["length"]=self.qubit_length

        pulse_args["style"] = style
        self.add_pulse(**pulse_args)

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        self.pulse(ch=cfg.qubit_ch, name="qubit_pulse", t=0)
        self.delay_auto(t=0.02, tag="waiting")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[cfg.adc_ch], pins=[0], t=cfg.device.readout.trig_offset[q])

    def make_pi_pulse(self, q, f, name):
        qubit_freq = f[q]
        pulse = {
            key: value[q] for key, value in self.cfg.device.qubit.pulses.pi_ge.items()
        }
        qubit_length = pulse.sigma * pulse.sigma_inc
        qubit_gain = pulse.gain
        qubit_ramp = pulse.sigma
        qubit_phase = 0
        pulse_args = {
            "ch": self.qubit_ch,
            "name": name,
            "length": qubit_length,
            "freq": qubit_freq,
            "phase": qubit_phase,
            "gain": qubit_gain,
        }

        if pulse.type == "gauss":
            style = "arb"
            self.add_gauss(
                ch=self.qubit_ch,
                name="ramp",
                sigma=qubit_ramp,
                length=qubit_length,
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"
        else:
            style = "const"

        pulse_args["style"] = style
        self.add_pulse(**pulse_args)


class LengthRabiProgram(QickProgram):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, cfg)

    def _initialize(self):
        cfg = AttrDict(self.cfg)
        self.frequency = cfg.expt.frequency[q]
        self.gain = cfg.expt.gain[q]
        self.readout_length = cfg.device.readout.readout_length[q]
        self.phase = cfg.device.readout.phase[q]

        self.qubit_freq = cfg.device.qubit.f_ge[q]
        self.qubit_length = cfg.device.qubit.pulses.pi_ge.sigma[q] * 4
        self.qubit_gain = cfg.device.qubit.pulses.pi_ge.gain[q]
        self.qubit_ramp = cfg.device.qubit.pulses.pi_ge.sigma[q]
        self.qubit_phase = 0

        super()._initialize()
        self.add_loop("freq_loop", self.cfg.expts)

        if self.cfg.expt.pulse_e:
            self.f_ef = cfg.device.qubit.f_ef[q]
            self.pi_ef_sigma = cfg.device.qubit.pulses.pi_ef.sigma[q]
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain[q]

    def _body(self):

        cfg = AttrDict(self.cfg)
        self.send_readout_config(ch=self.adc_ch, name="readout", t=0)
        if cfg.expt.pulse_e:
            self.pulse(ch=cfg.qubit_ch, name="qubit_pulse_e", t=0)
            self.delay_auto(t=0.01, tag="waiting")

        self.pulse(ch=cfg.qubit_ch, name="qubit_pulse", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=cfg.qubit_ch, name="qubit_pulse_e", t=0)
            self.delay_auto(t=0.01, tag="waiting")

        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="waiting 2")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[cfg.adc_ch], pins=[0], t=cfg.device.readout.trig_offset[q])
