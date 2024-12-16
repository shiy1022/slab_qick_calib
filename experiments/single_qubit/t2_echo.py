import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from datetime import datetime

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from qick_experiment import QickExperiment
import fitting as fitter

class RamseyEchoProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_wait = 3 # total wait time for each experiment
        self.r_phase2 = 4 # phase for the 2nd pi/2 pulse for each experiment
        if self.qubit_ch_type == 'int4':
            self.r_phase = self.sreg(self.qubit_ch, "freq")
            self.r_phase3 = 5 # for storing the left shifted value
        else: self.r_phase = self.sreg(self.qubit_ch, "phase")
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start/2/cfg.expt.num_pi))
        self.safe_regwi(self.q_rp, self.r_phase2, 0) 

        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = self.adc_ch
        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit_chan] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit_chan] = cfg.device.readout.gain
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma/2, gen_ch=self.qubit_ch)
        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi2_qubit", sigma=self.pi2sigma, length=self.pi2sigma*4)
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(-cfg.device.readout.phase, gen_ch = self.res_ch), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)

        # play pi/2 pulse with phase 0
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain, 
                waveform="pi2_qubit")
        else:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain, 
                length=self.pi2sigma)
        self.pulse(ch=self.qubit_ch)
        self.sync_all()

        for ii in range(cfg.expt.num_pi):
            # wait advanced wait time
            self.sync(self.q_rp, self.r_wait)

            if cfg.expt.cp: # pi pulse
                if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="arb",
                        freq=self.f_ge,
                        phase=0,
                        gain=cfg.device.qubit.pulses.pi_ge.gain, 
                        waveform="pi_qubit")
                else:
                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="const",
                        freq=self.f_ge,
                        phase=0,
                        gain=cfg.device.qubit.pulses.pi_ge.gain, 
                        length=self.pisigma)

            elif cfg.expt.cpmg: # pi pulse with phase pi/2
                if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="arb",
                        freq=self.f_ge,
                        phase=self.deg2reg(90, gen_ch=self.qubit_ch),
                        gain=cfg.device.qubit.pulses.pi_ge.gain, 
                        waveform="pi_qubit")
                else:
                    self.set_pulse_registers(
                        ch=self.qubit_ch,
                        style="const",
                        freq=self.f_ge,
                        phase=self.deg2reg(90, gen_ch=self.qubit_ch),
                        gain=cfg.device.qubit.pulses.pi_ge.gain, 
                        length=self.pisigma)
            else: assert False, 'Unsupported echo experiment type'
            self.pulse(ch=self.qubit_ch)

            # wait advanced wait time
            self.sync(self.q_rp, self.r_wait)

        # play pi/2 pulse with advanced phase
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="arb",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain, 
                waveform="pi2_qubit")
        else:
            self.set_pulse_registers(
                ch=self.qubit_ch,
                style="const",
                freq=self.f_ge,
                phase=0,
                gain=cfg.device.qubit.pulses.pi_ge.gain, 
                length=self.pi2sigma)
        if self.qubit_ch_type == 'int4':
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase3, '|', self.f_ge)
            self.mathi(self.q_rp, self.r_phase, self.r_phase3, "+", 0)
        else: self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_ch)

        # measure
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

    def update(self):
        # Update the wait time between each the pi pulses
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step/2/self.cfg.expt.num_pi))
        # Update the phase for the 2nd pi/2 pulse
        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.qubit_ch) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', phase_step)

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        shots_i0 = self.di_buf[0] / self.readout_length_adc #[self.cfg.expt.qubit]
        shots_q0 = self.dq_buf[0] / self.readout_length_adc #[self.cfg.expt.qubit]
        return shots_i0, shots_q0

class RamseyEchoExperiment(QickExperiment):
    """
    Ramsey Echo Experiment
    Experimental Config:
    expt = dict(
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        step: total wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        num_pi: number of pi pulses 
        cp: True/False
        cpmg: True/False
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, cfg_dict, prefix=None, progress=None, qi=0, go=True, params={}, style='', min_r2=None, max_err=None):
            #span=None, expts=100, ramsey_freq=0.1, reps=None, rounds=None,
        if prefix is None:
            prefix = f"echo_qubit{qi}"
            
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            'expts':100, 
            'ramsey_freq':0.1, 
            'span':3*self.cfg.device.qubit.T2e[qi], 
            'reps':2*self.reps, 
            'rounds':2*self.rounds, 
            'start':0.1, 
            'num_pi':1, 
            'cp':True, 
            'cpmg':False,
            'qubit':qi, 
            'qubit_chan':self.cfg.hw.soc.adcs.readout.ch[qi]}
        params = {**params_def, **params}    
        params['step'] = params['span']/params['expts']
        if params['ramsey_freq']=='smart':
            params['ramsey_freq'] = np.pi/2/self.cfg.device.qubit.T2e[qi]

        self.cfg.expt = params
        
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.update_config(q_ind=self.cfg.expt.qubit)    
        
        if self.cfg.expt.ramsey_freq >0: 
            self.cfg.expt.ramsey_freq_sign=1
        else:
            self.cfg.expt.ramsey_freq_sign=-1
        self.cfg.expt.ramsey_freq_abs = abs(self.cfg.expt.ramsey_freq)
        
        super().acquire(RamseyEchoProgram, progress=progress)
        
        return self.data

    def analyze(self, data=None, fit=True, debug=False, **kwargs):
        if data is None:
            data=self.data
        if fit:
            fitfunc = fitter.decaysin
            fitterfunc = fitter.fitdecaysin
            super().analyze(fitfunc, fitterfunc, data, **kwargs)
            data=self.data
            
            ydata_lab = ['amps', 'avgi', 'avgq']
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data['fit_'+ydata], (list, np.ndarray)): 
                    data['f_adjust_ramsey_'+ydata] = sorted((self.cfg.expt.ramsey_freq - data['fit_'+ydata][1], -self.cfg.expt.ramsey_freq - data['fit_'+ydata][1]), key=abs)              
                        
            fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(self.data, get_best_data_params=['f_adjust_ramsey'])
            
            f_pi_test = self.cfg.device.qubit.f_ge
            data['new_freq'] = f_pi_test + t2r_adjust[0]

        return data

    def display(self, data=None, fit=True, debug=False,plot_all=False,ax=None,savefig=True,show_hist=False, **kwargs):
        if data is None:
            data=self.data
        qubit = self.cfg.expt.qubit

        xlabel = "Wait Time ($\mu$s)"
        title=f"Ramsey Echo Q{qubit} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"
        fitfunc=fitter.decaysin
        captionStr = ['$T_2$ Echo : {val:.4} $\pm$ {err:.2g} $\mu$s','Freq. : {val:.3} $\pm$ {err:.1} MHz']
        var=[3,1]
        super().display(data=data,ax=ax,plot_all=plot_all,title=title, xlabel=xlabel, fit=fit, show_hist=show_hist,fitfunc=fitfunc,captionStr=captionStr,var=var)

        # # Plot the decaying exponential
        # x0 = -(p[2]+180)/360/p[1]
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], p[0], x0, p[3]), color='0.2', linestyle='--')
        # ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], -p[0], x0, p[3]), color='0.2', linestyle='--')


    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname