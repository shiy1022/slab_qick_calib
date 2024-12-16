import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from qick_experiment import QickExperiment, QickExperiment2D

import fitting as fitter

class PulseProbeSpectroscopyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.checkEF = cfg.expt.checkEF

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq") # get frequency register for qubit_ch 
        self.r_freq2 = 4
        if self.checkEF:
            self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_ch)
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_ch)
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)

        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
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
            
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)

        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start) # send start frequency to r_freq2

        # add pre-defined qubit and readout pulses to respective channels
        if self.checkEF:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=0, gain=cfg.device.readout.gain, length=self.readout_length_dac)

        self.synci(200)

    def body(self):
        cfg=AttrDict(self.cfg)

        # init to qubit excited state
        if self.checkEF: 
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")

        # setup and play ef probe pulse
        self.set_pulse_registers(
            ch=self.qubit_ch,
            style="const",
            freq=0, # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.qubit_ch)

        if self.checkEF: 
            # go back to ground state if in e to distinguish between e and f
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
        

class PulseProbeSpectroscopyExperiment(QickExperiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
        checkEF: flag to check EF transition
    )
    """

    def __init__(self, cfg_dict, prefix='', progress=None, qi=0, go=True, params={}, style='', checkEF=False, min_r2=None, max_err=None):
        
        prefix = 'qubit_spectroscopy_'
        if checkEF: 
            prefix = prefix+'ef' 
        prefix += style+f"_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
  
        # This one may need a bunch of options. 
        # coarse: wide span, medium gain, centered at ge freq
        # ef: coarse: medium span, extra high gain, centered at the ef frequency  
        # otherwise, narrow span, low gain, centered at ge frequency 
        max_len = 150 # Based on qick error messages, but not investigated 
        spec_gain = self.cfg.device.readout.spec_gain[qi]
        if style == 'coarse': 
            params_def = {'gain':1500*spec_gain, 'span':500, 'expts':500}
        elif style == 'fine':
            params_def = {'gain':100*spec_gain, 'span':5, 'expts':100}
        else: 
            params_def = {'gain':500*spec_gain, 'span':50, 'expts':200}
        if checkEF: 
            params_def['gain']=2*params_def['gain']
        params_def2={
            'relax_delay':10, 
            'len':50, 
            'reps':self.reps, 
            'rounds':self.rounds,
            'pulse_type':'const',
            'checkEF':checkEF, 
            'qubit':qi, 
            'qubit_chan':self.cfg.hw.soc.adcs.readout.ch[qi]}
        params_def = {**params_def, **params_def2}
        
        # combine params and params_Def, preferring params 
        params = {**params_def, **params}
        
        if checkEF: 
            params_def['start']=self.cfg.device.qubit.f_ef[qi]-params['span']/2
        else:
            params_def['start']=self.cfg.device.qubit.f_ge[qi]-params['span']/2
        params = {**params_def, **params}
        
        if params['length']=='t1' and not checkEF:
            params['length']=3*self.cfg.device.qubit.T1[qi]
        else:
            params['length']=self.cfg.device.qubit.T1[qi]/4
        if params['length']>max_len:
            params['length']=max_len
        params['step'] = params['span']/params['expts']
        
        self.cfg.expt = params
        
        if go: 
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        self.update_config(q_ind=q_ind)                  

        super().acquire(PulseProbeSpectroscopyProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data
        
        fitterfunc = fitter.fitlor
        fitfunc=fitter.lorfunc
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        data['new_freq']=data['best_fit'][2]
        return self.data

    def display(self, data=None, fit=True, signs=[1,1,1],ax=None,plot_all=True, **kwargs):
        if data is None:
            data=self.data 
        
        fitfunc=fitter.lorfunc
        xlabel = "Qubit Frequency (MHz)"
        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
        elif 'lo_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.lo_freq + data['xpts'][1:-1]
        else: 
            xpts = data['xpts'][1:-1]

        if self.cfg.expt.checkEF:
            title=f"EF Spectroscopy Q{self.cfg.expt.qubit} (Gain {self.cfg.expt.gain})"
        else:
            title=f"Spectroscopy Q{self.cfg.expt.qubit} (Gain {self.cfg.expt.gain})"
               
        captionStr = ['Freq: {val:.6} MHz', '$\kappa$: {val:.3} MHz']
        var = [2,3]
        super().display(data=data,ax=ax,plot_all=plot_all,title=title, xlabel=xlabel, fit=fit, show_hist=False,fitfunc=fitfunc,captionStr=captionStr,var=var)
                
    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname

    
class PulseProbePowerSweepSpectroscopyExperiment(QickExperiment2D):
    """
    self.cfg.expt: dict
        - start_f (float): Qubit frequency start [MHz].
        - step_f (float): Frequency step size [MHz].
        - expts_f (int): Number of experiments stepping from start frequency.
        - reps (int): Number of averages per point.
        - rounds (int): Number of start to finish sweeps to average over.
        - length (float): Qubit probe constant pulse length [us].
        - expts_gain (int): Number of gain experiments.
        - max_gain (int): Maximum gain for the sweep.
        - pulse_type (str): Type of pulse, default is 'const'.
        - checkEF (bool): Flag to check EF transition.
        - qubit (int): Qubit index.
        - qubit_chan (int): Qubit channel index.
        - relax_delay (float): Relaxation delay [us].
        - log (bool): Flag to indicate if logarithmic gain sweep is used.
        - rng (int): Range for logarithmic gain sweep.
    """
    

    def __init__(self, cfg_dict, prefix='', progress=None, qi=0, go=True, span=None, params={}, style='', checkEF=False,log=True, min_r2=None, max_err=None):
        
        prefix = 'qubit_spectroscopy_power_'
        if checkEF: 
            prefix = prefix+'ef' 
        prefix += style+f"_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
  
        max_len = 150 
        if style == 'coarse': 
            params_def = { 'span':800, 'expts':500}
        elif style == 'fine':
            params_def = {'span':40, 'expts':100}
        else: 
            params_def = { 'span':120, 'expts':200}
        if checkEF: 
            params_def['gain']=2*params_def['gain']
        params_def2={
            'relax_delay':10, 
            'length':25, 
            'reps':self.reps, 
            'rounds':self.rounds, 
            'rng':50, 
            'max_gain':self.cfg.device.qubit.max_gain, 
            'expts_gain':10,
            'pulse_type':'const', 
            'checkEF':checkEF, 
            'qubit':qi, 
            'qubit_chan':self.cfg.hw.soc.adcs.readout.ch[qi], 
            'log':log}
        params_def = {**params_def, **params_def2}
        
        # combine params and params_Def, preferreing params 
        params = {**params_def, **params}
        
        if checkEF: 
            params_def['start']=self.cfg.device.qubit.f_ef[qi]-params['span']/2
        else:
            params_def['start']=self.cfg.device.qubit.f_ge[qi]-params['span']/2
        params = {**params_def, **params}
        
        if params['length']=='t1' and not checkEF:
            params['length']=3*self.cfg.device.qubit.T1[qi]
        else:
            params['length']=self.cfg.device.qubit.T1[qi]/4
        if params['length']>max_len:
            params['length']=max_len
        params['step'] = params['span']/params['expts']
        
        self.cfg.expt = params
        if go: 
            self.go(progress=progress, display=True, analyze=True, save=True)

    def acquire(self, progress=False):
        super().update_config(q_ind=self.cfg.expt.qubit)        
        if 'log' in self.cfg.expt and self.cfg.expt.log==True:
            rng = self.cfg.expt.rng
            rat = rng**(-1/(self.cfg.expt["expts_gain"]-1))

            max_gain=self.cfg.expt['max_gain']
            gainpts = max_gain*rat**(np.arange(self.cfg.expt["expts_gain"]))
            gainpts = [int(g) for g in gainpts]
        else:
            gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        
        ysweep=[{'pts':gainpts, 'var':'gain'}]
        super().acquire(PulseProbeSpectroscopyProgram, ysweep, progress=progress)
        self.data["gainpts"] = gainpts
        return self.data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data
        
        fitfunc=fitter.lorfunc
        super().analyze(fitfunc)
            
        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):

        if self.cfg.expt.checkEF:
            title=f"EF Spectroscopy Power Sweep Q{self.cfg.expt.qubit}"
        else:
            title=f"Spectroscopy Power Sweep Q{self.cfg.expt.qubit}"
        
        xlabel = "Qubit Frequency (MHz)"
        ylabel = "Qubit Gain (DAC level)"

        super().display(data=data, ax=ax,plot_amps=plot_amps, title=title, xlabel=xlabel, ylabel=ylabel, fit=fit, **kwargs)
   
    def save_data(self, data=None):
        super().save_data(data=data)