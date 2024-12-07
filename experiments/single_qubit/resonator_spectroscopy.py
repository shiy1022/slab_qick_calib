import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, AttrDict
from scipy.signal import find_peaks

import experiments.fitting as fitter

"""
Measures the resonant frequency of the readout resonator when the qubit is in its ground state: sweep readout pulse frequency and look for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readouti.frequency.

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class ResonatorSpectroscopyProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        
        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type

        self.frequency = cfg.expt.frequency
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        self.res_gain = cfg.expt.gain
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

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
            mux_freqs[cfg.expt.qubit_chan] = self.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit_chan] = self.res_gain
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        # print(f'readout freq {mixer_freq} +/- {self.frequency}')
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=self.frequency, gen_ch=self.res_ch)

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain

        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f:
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.freqreg, phase=0, gain=self.res_gain, length=self.readout_length_dac)
        self.synci(200) # give processor some time to configure pulses

    def body(self):
        # pass
        cfg=AttrDict(self.cfg)
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
            self.sync_all() # align channels
        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            self.sync_all() # align channels

        
        self.measure(
            pulse_ch=self.res_ch,
            adcs=[self.adc_ch],
            adc_trig_offset=cfg.device.readout.trig_offset,
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

class ResonatorSpectroscopyExperiment(Experiment):
    """
    Resonator Spectroscopy Experiment
    Experimental Config
    expt = dict(
        start: start frequency (MHz), 
        step: frequency step (MHz), 
        expts: number of experiments, 
        pulse_e: boolean to add e pulse prior to measurement
        pulse_f: boolean to add f pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(self, soccfg=None, path='', prefix='ResonatorSpectroscopy', config_file=None, progress=None, im=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False):
        q_ind = self.cfg.expt.qubit
        if 'smart' in self.cfg.expt and self.cfg.expt.smart==True:
            kappa = np.abs(self.cfg.device.qubit['kappa'][q_ind])
            N = self.cfg.expt["expts"] 
            df = self.cfg.expt["step"]*N 
            w = df / kappa 
            at = np.arctan(2 *w/(1-w**2))+np.pi
            R = w / np.tan(at/2)
            fr = self.cfg.expt["start"]+df/2
            n = np.arange(N)-N/2
            xpts = fr + R * df / (2 * w)*np.tan(n/(N-1)*at)
        else:
            xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for f in tqdm(xpts, disable=not progress):
            self.cfg.expt.frequency = f
            rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            # print(rspec)
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
        
        for k, a in data.items():
            data[k]=np.array(a)
        
        self.data=data

        return data

    def analyze(self, data=None, fit=True, findpeaks=False, verbose=False, hanger=True, prom=20, **kwargs):
        if data is None:
            data=self.data
            
        if fit:                       
            if 'mixer_freq' in self.cfg.hw.soc.dacs.readout:
                xdata = self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'][1:-1]
            elif 'lo_freq' in self.cfg.hw.soc.dacs.readout:
                xdata = self.cfg.hw.soc.dacs.readout.lo_freq + data['xpts'][1:-1]
            else:
                xdata = data["xpts"][1:-1]


            ydata = data['amps'][1:-1]
            fitparams = [max(ydata), -(max(ydata)-min(ydata)), xdata[np.argmin(ydata)], 0.1 ]
            if hanger: 
                data['fit'], data['fit_err'], data['init'] = fitter.fithanger(xdata, ydata)
                r2 = fitter.get_r2(xdata, ydata, fitter.hangerS21func_sloped, data['fit']) 
                data['r2']=r2
                data['fit_err'] = np.mean(np.sqrt(np.diag(data['fit_err']))/np.abs(data['fit']))
#                if r2<0.5: 
#                    data['fit'] = [np.nan]*len(data['fit'])               
                if isinstance(data['fit'], (list, np.ndarray)):
                    f0, Qi, Qe, phi, scale, slope = data['fit']
                if 'lo' in self.cfg.hw:
                    print(float(self.cfg.hw.lo.readout.frequency)*1e-6)
                    print(f0)
                data['kappa']=f0*(1/Qi+1/Qe)*1e-4
                if verbose:
                    print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
                    print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
                    print('From fit:')
                    print(f'\tf0: {f0}')
                    print(f'\tQi: {Qi}')
                    print(f'\tQe: {Qe}')
                    print(f'\tQ0: {1/(1/Qi+1/Qe)}')
                    print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
                    print(f'\tphi [radians]: {phi}')
                if 'mixer_freq' in self.cfg.hw.soc.dacs.readout:
                    data['fit'][0]=data['fit'][0]-self.cfg.hw.soc.dacs.readout.mixer_freq
                if 'lo_freq' in self.cfg.hw.soc.dacs.readout:
                    data['fit'][0]=data['fit'][0]-self.cfg.hw.soc.dacs.readout.lo_freq

                
            else: 
                print(fitparams)
                data["lorentz_fit"]=fitter.fitlor(xdata, ydata, fitparams=fitparams)
                print('From Fit:')
                print(f'\tf0: {data["lorentz_fit"][2]}')
                print(f'\tkappa[MHz]: {data["lorentz_fit"][3]*2}')
        phs_data = np.unwrap(data["phases"][1:-1])
        #phs_fix=data['phases'][1:-1]
        slope, intercept = np.polyfit(data['xpts'][1:-1], phs_data, 1)        
        phs_fix = phs_data-slope*data["xpts"][1:-1]-intercept
        data['phase_fix'] = phs_fix
        if findpeaks: 
            xdata = data["xpts"][1:-1]
            ydata = data['amps'][1:-1]
            min_dist = 15 # minimum distance between peaks, may need to be edited if things are really close 
            max_width = 12 # maximum width of peaks in MHz, may need to be edited if peaks are off 
            df = xdata[1]-xdata[0]
            min_dist_inds = int(min_dist/df)
            max_width_inds = int(max_width/df)
            print(max_width_inds)

            #coarse_peaks = find_peaks(-ydata, distance=100, prominence= 2)#, width=3, threshold = 0.9, rel_height=1) 
            coarse_peaks, props = find_peaks(-ydata, distance=min_dist_inds, prominence=prom, width=[0,max_width_inds])#,threshold=0.2, rel_height=0.3)
            #coarse_peaks, props = find_peaks(ydata, distance=min_dist_inds, prominence=prom, width=[0,max_width_inds])

            data['coarse_peaks_index'] = coarse_peaks 
            data['coarse_peaks'] = xdata[coarse_peaks]
            #data['coarse_peaks_height'] = props['prominences']
            data['coarse_props']=props
        return data

    def display(self, data=None, fit=True, findpeaks=False, hanger=True, debug=True, ax=None, **kwargs):
        if data is None:
            data=self.data 
        
        if ax is not None:
            savefig = False
        else:
            savefig = True
        
        if 'lo' in self.cfg.hw:
            xpts = float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband[self.qubit]*(self.cfg.hw.soc.dacs.readout.mixer_freq[self.qubit] + data['xpts'][1:-1])       
        elif 'mixer_freq' in self.cfg.hw.soc.dacs.readout and fit:
            xpts = self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'][1:-1]      
            data['fit'][0]=data['fit'][0]+self.cfg.hw.soc.dacs.readout.mixer_freq
        elif 'lo_freq' in self.cfg.hw.soc.dacs.readout and fit:
            xpts = self.cfg.hw.soc.dacs.readout.lo_freq + data['xpts'][1:-1]
            data['fit'][0]=data['fit'][0]+self.cfg.hw.soc.dacs.readout.lo_freq
            data['init'][0]=data['init'][0]+self.cfg.hw.soc.dacs.readout.lo_freq
        else:
            xpts = data['xpts'][1:-1]
        qubit = self.cfg.expt.qubit
        title = f"Resonator Spectroscopy Q{qubit}, Gain {self.cfg.expt.gain}"
        if ax is None: 
            fig, ax = plt.subplots(2,1, figsize=(10,7))
        ax[0].set_ylabel("Amps [ADC units]")
        
        ax[0].plot(xpts, data['amps'][1:-1],'.-')
        if fit:
            if hanger:
                if not any(np.isnan(data["fit"])):
                    label=f"$\kappa$={data['kappa']:.2f} MHz"
                    ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *data["fit"]), label=label)               
                    ax[0].legend()
     
                if debug: 
                    ax[0].plot(xpts, fitter.hangerS21func_sloped(xpts, *data["init"]), label='Initial fit')
            elif not any(np.isnan(data["lorentz_fit"])):
                ax[0].plot(xpts, fitter.lorfunc(data["lorentz_fit"], xpts), label='Lorentzian fit')
            else:
                print("Lorentzian fit contains NaN values, skipping plot.")
        if findpeaks:
            num_peaks = len(data['coarse_peaks_index'])
            print('Number of peaks:', num_peaks)
            peak_indices = data['coarse_peaks_index']
            for i in range(num_peaks):
                peak = peak_indices[i]
                ax[0].axvline(xpts[peak], linestyle='--', color='0.2')
        
        if 'mixer_freq' in self.cfg.hw.soc.dacs.readout and fit:
            data['fit'][0]=data['fit'][0]-self.cfg.hw.soc.dacs.readout.mixer_freq
        elif 'lo_freq' in self.cfg.hw.soc.dacs.readout and fit:
            data['fit'][0]=data['fit'][0]-self.cfg.hw.soc.dacs.readout.lo_freq
            data['init'][0]=data['init'][0]-self.cfg.hw.soc.dacs.readout.lo_freq
        
        if savefig: 
            ax[1].set_xlabel("Readout Frequency [MHz]")
            ax[1].set_ylabel("Phase [radians]")
            ax[1].plot(xpts, data['phase_fix'],'.-')
        
            fig.tight_layout()
            imname = self.fname.split("\\")[-1]
            fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

class ResonatorPowerSweepSpectroscopyExperiment(Experiment):
    """Resonator Power Sweep Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start_f": start frequency (MHz), 
       "step_f": frequency step (MHz), 
       "expts_f": number of experiments in frequency,
       "start_gain": start frequency (dac units), 
       "step_gain": frequency step (dac units), 
       "expts_gain": number of experiments in gain sweep,
       "reps": number of reps, 
        } 
    """

    def __init__(self, soccfg=None, path='', prefix='ResonatorPowerSweepSpectroscopy', config_file=None, progress=None, im=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False):
        
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        
        if 'log' in self.cfg.expt and self.cfg.expt.log==True:
            rng = self.cfg.expt.rng
            rat = rng**(-1/(self.cfg.expt["expts_gain"]-1))

            max_gain = 32768
            gainpts = max_gain * rat**(np.arange(self.cfg.expt["expts_gain"]))
            gainpts = [int(g) for g in gainpts]
            rep_list = np.round(self.cfg.expt["reps"] * (1/rat**np.arange(self.cfg.expt["expts_gain"]))**2)
            print(rep_list)
            min_reps = 150
            for i in range(len(rep_list)):
                if rep_list[i]<min_reps:
                    rep_list[i]=min_reps
            
            #gainpts = self.cfg.expt["start_gain"] * 10**(self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"]))
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

        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for i in tqdm(pts, disable=not progress):
            self.cfg.expt.gain = gainpts[i]
            self.cfg.expt.reps = int(np.max([rep_list[i],35]))
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])

            for f in tqdm(xpts, disable=True):
                self.cfg.expt.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
        
        data["xpts"] = xpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data
        
        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        if fit:
            if highgain == None: highgain = np.max(data['gainpts'])
            if lowgain == None: lowgain = np.min(data['gainpts'])
            i_highgain = np.argmin(np.abs(data['gainpts']-highgain))
            i_lowgain = np.argmin(np.abs(data['gainpts']-lowgain))
            fit_highpow, err=fitter.fitlor(data["xpts"], data["amps"][i_highgain])
            fit_lowpow, err =fitter.fitlor(data["xpts"], data["amps"][i_lowgain])
            data['fit'] = [fit_highpow, fit_lowpow]
            data['fit_gains'] = [highgain, lowgain]
            data['lamb_shift'] = fit_highpow[2] - fit_lowpow[2]
        
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        qubit = self.cfg.expt.qubit
        inner_sweep = data['xpts'] #float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'])
        outer_sweep = data['gainpts']

        amps = data['amps']
        for i in range(len(amps)):
            amps[i,:] =amps[i,:]/np.median(amps[i,:])
        
        y_sweep = outer_sweep
        x_sweep = inner_sweep

        # if 'log' in self.cfg.expt and self.cfg.expt.log:
        #     y_sweep = np.log10(y_sweep)

        # THIS IS CORRECT EXTENT LIMITS FOR 2D PLOTS
        fig, ax =plt.subplots(1,1,figsize=(10,8))
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap='viridis', shading='auto')
        if 'log' in self.cfg.expt and self.cfg.expt.log:
            plt.yscale('log')
        if fit:
            fit_highpow, fit_lowpow = data['fit']
            highgain, lowgain = data['fit_gains']
            plt.axvline(fit_highpow[2], linewidth=1, color='0.2')
            plt.axvline(fit_lowpow[2], linewidth=1, color='0.2')
            plt.plot(x_sweep, [highgain]*len(x_sweep), linewidth=1, color='0.2')
            plt.plot(x_sweep, [lowgain]*len(x_sweep), linewidth=1, color='0.2')
            print(f'High power peak [MHz]: {fit_highpow[2]}')
            print(f'Low power peak [MHz]: {fit_lowpow[2]}')
            print(f'Lamb shift [MHz]: {data["lamb_shift"]}')
            
        plt.title(f"Resonator Spectroscopy Power Sweep Q{qubit}")
        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Resonator Gain [DAC level]")

        ax.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True, right=True, labelright=False)

        # plt.clim(vmin=-0.2, vmax=0.2)
        #plt.clim(vmin=-10, vmax=5)
        plt.colorbar(label='Amps/Avg [ADC level]')
        plt.show()
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

class Resonator2DSpectroscopyExperiment(Experiment):
    """Resonator 2D Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start_f": start frequency (MHz), 
       "step_f": frequency step (MHz), 
       "expts_f": number of experiments in frequency,
       "reps": number of reps, 
        } 
    """

    def __init__(self, soccfg=None, path='', prefix='Resonator2DSpectroscopy', config_file=None, progress=None, im=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False):
       
        xpts = self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])
        pts = np.arange(self.cfg.expt['pts'])
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

        data={"xpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[],'pts':pts}
        for i in tqdm(pts, disable=not progress):
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            

            for f in tqdm(xpts, disable=True):
                self.cfg.expt.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
        
        data["xpts"] = xpts
        
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

        inner_sweep = data['xpts'] #float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + data['xpts'])
        outer_sweep = data['pts']
        
        y_sweep = outer_sweep
        x_sweep = inner_sweep

        # THIS IS CORRECT EXTENT LIMITS FOR 2D PLOTS
        fig=plt.figure(figsize=(10,8))
        plt.pcolormesh(x_sweep, y_sweep, data['amps'], cmap='viridis', shading='auto')
        
        plt.title(f"Resonator Spectroscopy Time Sweep")
        plt.xlabel("Resonator Frequency [MHz]")
        plt.ylabel("Time")
        # plt.clim(vmin=-0.2, vmax=0.2)
        #plt.clim(vmin=-10, vmax=5)
        plt.colorbar(label='Amps/Avg [ADC level]')
        plt.show()
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

class ResonatorVoltSweepSpectroscopyExperiment(Experiment):
    """Resonator Volt Sweep Spectroscopy Experiment
       Experimental Config
       expt_cfg={
       "start_f": start frequency (MHz), 
       "step_f": frequency step (MHz), 
       "expts_f": number of experiments in frequency,
       "start_volt": start volt, 
       "step_volt": voltage step, 
       "expts_volt": number of experiments in voltage sweep,
       "reps": number of reps, 
       "dc_ch": channel on dc_instr to sweep voltage
        } 
    """

    def __init__(self, soccfg=None, path='', dc_instr=None, dc_ch=None, prefix='ResonatorVoltSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.dc_instr = dc_instr

    def acquire(self, progress=False):
        xpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        voltpts = self.cfg.expt["start_volt"] + self.cfg.expt["step_volt"]*np.arange(self.cfg.expt["expts_volt"])
        
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

        data={"xpts":[], "voltpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}

        self.dc_instr.set_mode('CURR')
        self.dc_instr.set_current_limit(max(abs(voltpts)*5))
        print(f'Setting current limit {self.dc_instr.get_current_limit()*1e6} uA')
        self.dc_instr.set_output(True)

        for volt in tqdm(voltpts, disable=not progress):
            # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=volt)
            self.dc_instr.set_current(volt)
            print(f'current set to {self.dc_instr.get_current() * 1e6} uA')
            time.sleep(0.5)
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            
            for f in tqdm(xpts, disable=True):
                self.cfg.device.readout.frequency = f
                rspec = ResonatorSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                self.prog = rspec
                avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
                phase = np.angle(avgi+1j*avgq) # Calculating the phase
                
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phase)
            time.sleep(0.5)
        # self.dc_instr.initialize()
        # self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=0)

        self.dc_instr.set_current(0)
        print(f'current set to {self.dc_instr.get_current() * 1e6} uA')
        
        data["xpts"] = xpts
        data["voltpts"] = voltpts
        
        for k, a in data.items():
            data[k] = np.array(a)
        
        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        plt.figure(figsize=(12, 8))
        x_sweep = 1e3*data['voltpts']
        y_sweep = data['xpts']
        amps = data['amps']
        # for amps_volt in amps:
        #     amps_volt -= np.average(amps_volt)
        
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(data['amps']), 0), cmap='viridis')
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                plt.pcolormesh(
                    1e3*add_data['voltpts'], add_data['xpts'], np.flip(np.rot90(add_data['amps']), 0), cmap='viridis')
        if fit: pass
            
        plt.title(f"Resonator {self.cfg.expt.qubit} sweeping DAC box ch {self.cfg.expt.dc_ch}")
        plt.ylabel("Resonator frequency")
        plt.xlabel("DC current [mA]")
        # plt.ylabel("DC voltage [V]")
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')

        # plt.plot(x_sweep, amps[1])
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname