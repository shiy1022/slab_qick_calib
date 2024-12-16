import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import fitting as fitter
from qick_experiment import QickExperiment, QickExperiment2D

class T1Program(RAveragerProgram):
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
        self.r_wait = 3
        self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
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

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)

        # add qubit and readout pulses to respective channels
        if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
            self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        else:
            self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(-self.cfg.device.readout.phase, gen_ch = self.res_ch), gain=cfg.device.readout.gain, length=self.readout_length_dac)


        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        self.pulse(ch=self.qubit_ch)
        self.sync_all() # align channels
        self.sync(self.q_rp, self.r_wait) # wait for the time stored in the wait variable register
        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
        
    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        shots_i0 = self.di_buf[0] / self.readout_length_adc #[self.cfg.expt.qubit]
        shots_q0 = self.dq_buf[0] / self.readout_length_adc #[self.cfg.expt.qubit]
        return shots_i0, shots_q0
    
    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time


class T1Experiment(QickExperiment):
    """
    self.cfg.expt: dict
        A dictionary containing the configuration parameters for the T1 experiment. The keys and their descriptions are as follows:
        - start (int): The initial wait time between the two pi/2 pulses in microseconds.
        - span (float): The total span of the wait time sweep in microseconds.
        - step (float): The step size for the wait time increments in microseconds.
        - expts (int): The number of experiments to be performed.
        - reps (int): The number of repetitions for each experiment.
        - rounds (int): The number of rounds for the experiment.
        - qubit (int): The index of the qubit being used in the experiment.
        - qubit_chan (int): The channel of the qubit being read out.
    """
    

    def __init__(self, cfg_dict,  qi=0, go=True, params={},prefix=None, progress=None, style='', min_r2=None, max_err=None):

        if prefix is None:
            prefix = f"t1_qubit{qi}"
            
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            'expts':60,  
            'span':3.7*self.cfg.device.qubit.T1[qi], 
            'reps':2*self.reps, 
            'rounds':self.rounds, 
            'start':0,'qubit':qi, 
            'qubit_chan':self.cfg.hw.soc.adcs.readout.ch[qi]}
        
        if style=='fine': 
            params_def['rounds'] = params_def['rounds']*2
        elif style=='fast':
            params_def['expts'] = 30
        
        params = {**params_def, **params}     
        params['step'] = params['span']/params['expts']
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)
            
    def acquire(self, progress=False, debug=False):
        
        q_ind = self.cfg.expt.qubit
        self.update_config(q_ind=q_ind)                           
        super().acquire(T1Program, progress=progress)
        
        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        fitfunc = fitter.expfunc
        fitterfunc=fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        data['new_t1']=data['best_fit'][2]
        data['new_t1_i']=data['fit_avgi'][2]
        return data

    def display(self, data=None, fit=True,plot_all=False,ax=None, show_hist=False, **kwargs):
        qubit=self.cfg.expt.qubit
        title=f'$T_1$ Q{qubit}'
        xlabel = "Wait Time ($\mu$s)"
        captionStr = ['$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s']
        var=[2]
        fitfunc = fitter.expfunc
        
        super().display(data=data,ax=ax,plot_all=plot_all,title=title, xlabel=xlabel, fit=fit, show_hist=show_hist,fitfunc=fitfunc,captionStr=captionStr,var=var)
        
    def save_data(self, data=None):
        super().save_data(data=data)
        return self.fname
    

class T1Continuous(QickExperiment):
    """
    T1 Continuous
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """
    def __init__(self, soccfg=None, path='', prefix='T1Continuous', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)
    
    def acquire(self, progress=False, debug=False):

        self.update_config(q_ind=self.cfg.expt.qubit)      
        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        shots_i, shots_q = t1.collect_shots()

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase    

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_time = current_time.encode('ascii','replace')

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases, 'time':current_time, 'raw_i': shots_i, 'raw_q': shots_q, 'raw_amps': np.abs(shots_i+1j*shots_q)}   
        
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        pass
                
        
    def display(self, data=None, fit=True, show = False, **kwargs):
        pass

    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    

class T1_2D(QickExperiment2D):
    """
    sweep_pts = number of points in the 2D sweep
    """
    
    def __init__(self, cfg_dict, qi=0, go=True, params={},prefix=None, progress=None, style='', min_r2=None, max_err=None):

        if prefix is None:
            prefix = f"t1_2d_qubit{qi}"
            
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            'expts':60,  
            'span':3.7*self.cfg.device.qubit.T1[qi], 
            'reps':2*self.reps, 
            'rounds':self.rounds, 
            'start':0, 
            'sweep_pts':200,
            'qubit':qi, 
            'qubit_chan':self.cfg.hw.soc.adcs.readout.ch[qi]}
        if style=='fine': 
            params_def['rounds'] = params_def['rounds']*2
        elif style=='fast':
            params_def['expts'] = 30
        params = {**params_def, **params}    
    
        params['step'] = params['span']/params['expts']
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)
    
    
    def acquire(self, progress=False, debug=False):

        super().update_config(q_ind=self.cfg.expt.qubit)              
        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{'pts':sweep_pts, 'var':'count'}]
        super().acquire(T1Program, y_sweep, progress=progress)

        return self.data
    
    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        
        fitfunc = fitter.expfunc
        fitterfunc=fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)

    def display(self, data=None, fit=True,ax=None, **kwargs):
        if data is None:
            data=self.data 
        
        title = f'$T_1$ 2D Q{self.cfg.expt.qubit}'
        xlabel = f'Wait Time ($\mu$s)'
        ylabel = 'Time (s)'

        super().display(data=data, ax=ax,title=title, xlabel=xlabel, ylabel=ylabel, fit=fit, **kwargs)

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname