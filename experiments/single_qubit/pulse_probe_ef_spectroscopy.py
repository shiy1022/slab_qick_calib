import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class PulseProbeEFSpectroscopyProgram(RAveragerProgram):
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
        

class PulseProbeEFSpectroscopyExperiment(Experiment):
    """
    PulseProbe EF Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEFSpectroscopy', config_file=None, progress=None,im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

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

        qspec_ef=PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, )        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data
        
        if fit:
            fitterfunc = fitter.fitlor
            fitfunc=fitter.lorfunc
            ydata_lab = ['amps', 'avgi', 'avgq']
            for i, ydata in enumerate(ydata_lab):
                data['fit_' + ydata], data['fit_err_' + ydata] = fitterfunc(data['xpts'], data[ydata])
            #data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
            
            fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitter.lorfunc)
            r2 = fitter.get_r2(data['xpts'],data[i_best], fitfunc, fit_pars)
            print('R2:', r2)
            data['r2']=r2
            data['best_fit']=fit_pars
            print('Best fit:', i_best)
            data['fit_err']=np.mean(np.abs(fit_err/fit_pars))
            print('fit_err:', data['fit_err'])

            i_best = i_best.encode("ascii", "ignore")
            data['i_best']=i_best
        return data

    def display(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data 

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts']
        elif 'lo_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.lo_freq + data['xpts'][1:-1]
        else: 
            xpts = data['xpts']
        if self.cfg.expt.checkEF:
            title=f"EF Spectroscopy Q{self.cfg.expt.qubit} (Gain {self.cfg.expt.gain})"
        else:
            title=f"Spectroscopy Q{self.cfg.expt.qubit} (Gain {self.cfg.expt.gain})"
        fitfunc=fitter.lorfunc
        fig, ax=plt.subplots(3, 1, figsize=(9, 10))
        xlabel = "Wait Time (us)"
        ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
        fig.suptitle(title)
        ydata_lab = ['amps', 'avgi', 'avgq']
        for i, ydata in enumerate(ydata_lab):
            ax[i].plot(xpts, data[ydata],'o-')
        
            if fit:
                p = data['fit_'+ydata]
                pCov = data['fit_err_amps']
                captionStr = f'Freq: fit [MHz]: {p[2]:.3} $\pm$ {np.sqrt(pCov[2][2]):.3} \n'
                captionStr += f'Kappa [MHz]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3} \n'
                ax[i].plot(data["xpts"], fitfunc(data["xpts"], *p), label=captionStr)
                ax[i].set_ylabel(ylabels[i])
                ax[i].set_xlabel(xlabel)
                ax[i].legend()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


class PulseProbeEFPowerSweepSpectroscopyExperiment(Experiment):
    
    """
    Pulse probe EF power sweep spectroscopy experiment
    Experimental Config
        expt = dict(
        start_f: start ef probe frequency [MHz]
        step_f: step ef probe frequency
        expts_f: number experiments freq stepping from start
        start_gain: start ef const pulse gain (dac units)
        step_gain
        expts_gain
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEFPowerSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        
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
       
        data={"fpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for gain in tqdm(gainpts):
            self.cfg.expt.gain = gain
            self.cfg.expt.start = self.cfg.expt.start_f
            self.cfg.expt.step = self.cfg.expt.step_f
            self.cfg.expt.expts = self.cfg.expt.expts_f
            spec = PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = spec
            x_pts, avgi, avgq = spec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        data["fpts"] = fpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        x_sweep = data['fpts']
        y_sweep = data['gainpts'] 
        avgi = data['avgi']
        avgq = data['avgq']
        for avgi_gain in avgi:
            avgi_gain -= np.average(avgi_gain)
        for avgq_gain in avgq:
            avgq_gain -= np.average(avgq_gain)


        fig=plt.figure(figsize=(10,12))
        plt.subplot(211, title="Pulse Probe EF Spectroscopy Power Sweep", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps-Avg [adc level]')

        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Phases-Avg [radians]')

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
        plt.show()    

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
    
class PulseProbePowerSweepSpectroscopyExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
        start_f: Qubit frequency [MHz]
        step_f
        expts_f: Number of experiments stepping from start
        "start_gain": start frequency (dac units), 
       "step_gain": frequency step (dac units), 
       "expts_gain": number of experiments in gain sweep,
        reps: Number of averages per point
        rounds: Number of start to finish sweeps to average over
        length: Qubit probe constant pulse length [us]
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbePowerSweepSpectroscopy', config_file=None, progress=None, im=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])    
        if 'log' in self.cfg.expt and self.cfg.expt.log==True:

            rng = self.cfg.expt.rng
            rat = rng**(-1/(self.cfg.expt["expts_gain"]-1))

            max_gain=32768
            gainpts = max_gain*rat**(np.arange(self.cfg.expt["expts_gain"]))
            gainpts = [int(g) for g in gainpts]
        else:
            gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        rep_list = self.cfg.expt["reps"] * np.ones(self.cfg.expt["expts_gain"])
        pts = np.arange(self.cfg.expt["expts_gain"])
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

        self.cfg.expt.start = self.cfg.expt.start_f
        self.cfg.expt.step = self.cfg.expt.step_f
        self.cfg.expt.expts = self.cfg.expt.expts_f
        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(pts, disable=not progress):
            self.cfg.expt.gain = gainpts[i]
            self.cfg.expt.reps = int(rep_list[i])
            # data["avgi"].append([])
            # data["avgq"].append([])
            # data["amps"].append([])
            # data["phases"].append([])

            qspec = PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False)                        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        data["xpts"] = xpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data
        
        par_list =[]
        # Lorentzian fit at 
        if fit:
            for i in range(len(data['gainpts'])):
                try:
                    fit_pars, fit_errs=fitter.fitlor(data["xpts"], data["amps"][i])
                    par_list.append(fit_pars)
                except:
                    par_list.append([np.nan, np.nan, np.nan, np.nan])
            freqs = [par[2] for par in par_list]
            data['freqs']=freqs
            print(data['freqs'])
            
        
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        if self.cfg.expt.checkEF:
            title=f"EF Spectroscopy Power Sweep Q{self.cfg.expt.qubit}"
        else:
            title=f"Spectroscopy Power Sweep Q{self.cfg.expt.qubit}"

        inner_sweep = data['xpts']
        outer_sweep = data['gainpts']

        amps = data['amps']

        y_sweep = outer_sweep
        x_sweep = inner_sweep

        fig=plt.figure(figsize=(10,8))
        plt.title(title)
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap='viridis', shading='auto')
        if 'log' in self.cfg.expt and self.cfg.expt.log:
            plt.yscale('log')
        plt.xlabel("Qubit Frequency [MHz]")
        plt.ylabel("Qubit Gain [DAC level]")
        
        plt.colorbar(label='Amps/Avg [ADC level]')
        plt.show()
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)