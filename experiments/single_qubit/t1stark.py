import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import copy
import experiments.fitting as fitter

class T1StarkProgram(AveragerProgram):
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

        #self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
        #self.r_wait = 3
        #self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
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
        #    self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        else:
            self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(-self.cfg.device.readout.phase, gen_ch = self.res_ch), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        if self.cfg.expt.acStark: 
            self.stark_freq = self.freq2reg(cfg.expt.stark_freq, gen_ch=self.qubit_ch)
            self.stark_gain = self.cfg.expt.stark_gain # gain of the pulse we are trying to calibrate
        self.stark_length = self.us2cycles(cfg.expt.length)
            

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
        self.pulse(ch=self.qubit_ch)
        #self.sync_all() # align channels
        if self.cfg.expt.acStark:
            #self.stark_time = self.us2cycles(self.r_wait, gen_ch=self.qubit_chs[qTest])
            self.set_pulse_registers(
                    ch=self.qubit_ch,
                    style="const",
                    freq=self.stark_freq,
                    phase=0,
                    gain=self.stark_gain, # gain set by update
                    length=self.stark_length)
            self.pulse(ch=self.qubit_ch)
            self.sync_all(5) # align channels and wait 50ns
        else:
            self.sync_all(self.stark_length) # align channels and wait 50ns
        #self.wait_all(self.stark_length) # wait for the pulse to finish

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



class T1StarkExperiment(Experiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1_stark', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        
       
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                

                                
        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        xvals =  np.arange(self.cfg.expt["expts"])
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for i in tqdm(xvals, disable=not progress):
            length = lengths[i]
            self.cfg.expt.length = float(length)
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            #print(t1)
            self.prog = t1
            avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        for k, a in data.items():
            data[k]=np.array(a)  
        
        current_time = current_time.encode('ascii','replace')
        data['time']=current_time
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        fitfunc = fitter.expfunc
        fitterfunc=fitter.fitexp
        ydata_lab = ['amps', 'avgi', 'avgq']
        for i, ydata in enumerate(ydata_lab):
            data['fit_' + ydata], data['fit_err_' + ydata] = fitterfunc(data['xpts'], data[ydata], fitparams=None)

        fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitfunc)
        r2 = fitter.get_r2(data['xpts'],data[i_best], fitfunc, fit_pars)
        print('R2:', r2)
        data['r2']=r2

        data['fit_err']=np.mean(np.abs(fit_err/fit_pars))
        print('fit_err:', data['fit_err'])
        
        data['best_fit']=fit_pars
        data['new_t1']=fit_pars[2]
        print('Best fit:', i_best)
        i_best = i_best.encode("ascii", "ignore")
        data['i_best']=i_best

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        qubit = self.cfg.expt.qubit
        
        fig, ax=plt.subplots(3, 1, figsize=(9, 10))
        xlabel = "Wait Time (us)"
        ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
        fig.suptitle(f'T1 Stark Q{qubit}')
        ydata_lab = ['amps', 'avgi', 'avgq']
        for i, ydata in enumerate(ydata_lab):
            ax[i].plot(data["xpts"], data[ydata],'o-')
        
            if fit:
                p = data['fit_'+ydata]
                pCov = data['fit_err_amps']
                captionStr = f'$T_1$ fit [us]: {p[2]:.3} $\pm$ {np.sqrt(pCov[2][2]):.3}'
                ax[i].plot(data["xpts"], fitter.expfunc(data["xpts"], *p), label=captionStr)
                ax[i].set_ylabel(ylabels[i])
                ax[i].set_xlabel(xlabel)
                ax[i].legend(loc='upper right')

        if fit: 
            data["err_ratio_amps"] = np.sqrt(data['fit_err_amps'][2][2])/data['fit_amps'][2]
            data["err_ratio_i"] = np.sqrt(data['fit_err_avgi'][2][2])/data['fit_avgi'][2] 
            data["err_ratio_q"] = np.sqrt(data['fit_err_avgq'][2][2])/data['fit_avgq'][2]

        imname = self.fname.split("\\")[-1]
        fig.tight_layout()
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    
# ====================================================== #

class T1Continuous(Experiment):
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
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                
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
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
                
        return data

        
    def display(self, data=None, fit=True, show = False, **kwargs):
        if data is None:
            data=self.data 
    
        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"], data["avgi"],'o-', label = 'Current Data')
        plt.plot(self.cfg.expt.prev_data_x, self.cfg.expt.prev_data_i,'o-', label = 'Previous Data')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
            data["err_ratio_i"] = np.sqrt(data['fit_err_avgi'][3][3])/data['fit_avgi'][3]
        plt.legend()
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"], data["avgq"],'o-', label = 'Current Data')
        plt.plot(self.cfg.expt.prev_data_x, self.cfg.expt.prev_data_q,'o-', label = 'Previous Data')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
            data["err_ratio_q"] = np.sqrt(data['fit_err_avgq'][3][3])/data['fit_avgq'][3]

        plt.legend()
        if show:
            plt.show() 

    
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    

class T1_2D(Experiment):
    """
    sweep_pts = number of points in the 2D sweep
    """
    def __init__(self, soccfg=None, path='', prefix='T1_2D', config_file=None, progress=None, im=None):
            super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)
    
    def acquire(self, progress=False, debug=False):

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode('ascii','replace')

        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})    
                    
        sweeppts = np.arange(self.cfg.expt["sweep_pts"])
        data={"xpts":[], "sweeppts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        for i in tqdm(sweeppts):
            t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        
            xpts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)
        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)
        
        data['xpts'] = xpts
        data['sweeppts'] = sweeppts
        data['time'] = current_time

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data
    
    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['xpts']
        y_sweep = data['sweeppts']
        avgi = data['avgi']
        avgq = data['avgq']

        plt.figure(figsize=(10,8))
        plt.subplot(211, title="T1 2D", ylabel="Points")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I [ADC level]')
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, xlabel="Points", ylabel="Frequency [MHz]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()

        plt.plot(x_sweep, data['amps'][0])
        plt.title(f'First point')
        plt.show()

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
        
class T1StarkPowerExperiment(Experiment):
    """
    Stark Power Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz), 
        step_f: frequency step (MHz), 
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='RamseyStarkFreq', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})     


        span_gain = self.cfg.expt["expts_gain"]*self.cfg.expt["step_gain"]
        coef = span_gain / np.sqrt(self.cfg.expt["expts_gain"])
        gainpts = self.cfg.expt["start_gain"] + coef*np.sqrt(np.arange(self.cfg.expt["expts_gain"]))
        gainpts = gainpts.astype(int)
        data={"xpts":[], "freqpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        xvals =  np.arange(self.cfg.expt["expts"])

        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])


        for gain in tqdm(gainpts):
            self.cfg.expt.stark_gain = gain
            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for i in range(len(xvals)):
                length = lengths[i]
                self.cfg.expt.length = float(length)
                
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                #print(ramsey)
                self.prog = t1
                avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["xpts"][-1].append(length)
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)

        

        data['gainpts'] = gainpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        
        fitfunc = fitter.expfunc
        fitterfunc=fitter.fitexp
        ydata_lab = ['amps', 'avgi', 'avgq']
        ydata_lab= ['avgi']
        for i, ydata in enumerate(ydata_lab):
            data['fit_'+ydata] = []
            for i in range(len(data['gainpts'])):
                fit_pars = []
                #data['fit_' + ydata], data['fit_err_' + ydata] = fitterfunc(data['xpts'], data[ydata], fitparams=None)
                fit_pars, fit_err = fitterfunc(data['xpts'][i], data[ydata][i], fitparams=None)
                data['fit_'+ydata].append(fit_pars)
        t1_fits = [data['fit_avgi'][i][2] for i in range(len(data['gainpts']))]
        data['t1_fits'] = t1_fits



    def display(self, data=None, fit=True,plot_both=False, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['xpts']
        y_sweep = data['gainpts']**2/np.max(data['gainpts'])**2
        avgi = data['avgi']
        avgq = data['avgq']
        title = f'Amplitude Stark T1, Frequency: {self.cfg.expt.stark_freq-self.cfg.device.qubit.f_ge} MHz' 
        if plot_both: 
            fig=plt.figure(figsize=(10,8))
            plt.subplot(211, title=title, ylabel="Gain [DAC units]")
            plt.pcolormesh(x_sweep, y_sweep, avgi, cmap='viridis', shading='auto')
            plt.colorbar(label='I [ADC level]')
            plt.clim(vmin=None, vmax=None)

            plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Amplitude [MHz]")
            plt.pcolormesh(x_sweep, y_sweep, avgq, cmap='viridis', shading='auto')

            plt.colorbar(label='Q [ADC level]')
            plt.clim(vmin=None, vmax=None)
            
            plt.tight_layout()
            plt.show()
        else:
            fig=plt.figure(figsize=(7,8))
            plt.title(title)
            plt.pcolormesh(x_sweep, y_sweep, avgi, cmap='viridis', shading='auto')
            plt.colorbar(label='I [ADC level]')
            plt.clim(vmin=None, vmax=None)
            plt.ylabel('Gain [DAC units]')

            plt.tight_layout()
            plt.show()

        
        fig2=plt.figure()
        plt.plot(data['gainpts']**2/np.max(data['gainpts']**2), data['t1_fits'])
        plt.xlabel('Gain Sq')
        plt.ylabel('T1 (us)')

        plt.figure()
        for i in range(len(data['gainpts'])):
            plt.plot(data['xpts'][i], data['avgi'][i]+i, label=f'Gain {data["gainpts"][i]}')        
        
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

        imname = self.fname.split("\\")[-1]
        fig2.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'_t1.png')
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    

class T1StarkPowerContExperiment(Experiment):
    """
    Stark Power Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz), 
        step_f: frequency step (MHz), 
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1StarkPowerCont', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})     


        span_gain = self.cfg.expt.stop_gain - self.cfg.expt.start_gain
        coef = span_gain / np.sqrt(self.cfg.expt["expts"])
        gainpts = self.cfg.expt["start_gain"] + coef*np.sqrt(np.arange(self.cfg.expt["expts"]))
        gainpts = gainpts.astype(int)
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], 'avgi_off':[], 'avgq_off':[], 'amps_off':[], 'phases_off':[]}

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1

        self.cfg.Eexpt.length=0
        self.cfg.T1expt.length=self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False

        for gain in tqdm(gainpts):

            self.cfg.expt=copy.deepcopy(self.cfg.Eexpt)
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase
            data["avgi_off"].append(avgi)
            data["avgq_off"].append(avgq)
            data["amps_off"].append(amp)
            data["phases_off"].append(phases)

            self.cfg.expt=copy.deepcopy(self.cfg.T1expt)
            self.cfg.expt.stark_gain = gain       
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(gain)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phases)
  
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        pass



    def display(self, data=None, fit=True,plot_both=False, **kwargs):
        if data is None:
            data=self.data 
        
        title = f'Amplitude Stark T1, Frequency: {self.cfg.expt.stark_freq-self.cfg.device.qubit.f_ge} MHz' 
    
        fig=plt.figure(figsize=(7,8))
        plt.title(title)
        plt.plot(data['xpts']**2/np.max(data['xpts']**2),data['avgi']/data['avgi_off'], label='I')
        plt.xlabel('Gain Sq[DAC units]')

        plt.tight_layout()
        plt.show()

        
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    

class T1StarkPowerContTimeExperiment(Experiment):
    """
    Stark Power Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz), 
        step_f: frequency step (MHz), 
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1StarkPowerCont', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})     


        span_gain = self.cfg.expt.stop_gain - self.cfg.expt.start_gain
        coef = span_gain / np.sqrt(self.cfg.expt["expts"])
        gainpts = self.cfg.expt["start_gain"] + coef*np.sqrt(np.arange(self.cfg.expt["expts"]))
        gainpts = gainpts.astype(int)
        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], 'avgi_off':[], 'avgq_off':[], 'amps_off':[], 'phases_off':[]}

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1

        self.cfg.Eexpt.length=0
        self.cfg.T1expt.length=self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False
        for tm in tqdm(np.arange(self.cfg.expt.ntimes)):
            data["avgi_off"].append([])
            data["avgq_off"].append([])
            data["amps_off"].append([])
            data["phases_off"].append([])
            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for gain in gainpts:

                self.cfg.expt=copy.deepcopy(self.cfg.Eexpt)
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phases = np.angle(avgi+1j*avgq) # Calculating the phase
                data["avgi_off"][-1].append(avgi)
                data["avgq_off"][-1].append(avgq)
                data["amps_off"][-1].append(amp)
                data["phases_off"][-1].append(phases)

                self.cfg.expt=copy.deepcopy(self.cfg.T1expt)
                self.cfg.expt.stark_gain = gain       
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)        
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phases = np.angle(avgi+1j*avgq) # Calculating the phase
                data["xpts"][-1].append(gain)
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phases)
  
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data

        pass



    def display(self, data=None, fit=True,plot_both=False, **kwargs):
        if data is None:
            data=self.data 
        
        title = f'Amplitude Stark T1, Frequency: {self.cfg.expt.stark_freq-self.cfg.device.qubit.f_ge} MHz' 
        y_sweep = np.arange(self.cfg.expt.ntimes)
        fig=plt.figure(figsize=(7,8))
        plt.title(title)
        plt.pcolormesh(data['xpts']**2/np.max(data['xpts']**2),y_sweep, data['avgi']/data['avgi_off'], label='I')
        plt.colorbar()
        plt.xlabel('Gain Sq[DAC units]')

        plt.tight_layout()
        plt.show()

        
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname