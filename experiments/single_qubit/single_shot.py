import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from copy import deepcopy

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

blue="#4053d3"
red ="#b51d14"
int_rgain=True
def hist(data, plot=True, span=None, verbose=True):

    """
    span: histogram limit is the mean +/- span
    """
    Ig = data['Ig']
    Qg = data['Qg']
    Ie = data['Ie']
    Qe = data['Qe']
    plot_f = False 
    if 'If' in data.keys():
        plot_f = True
        If = data['If']
        Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f: xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        fig.tight_layout()
        m=0.7
        a=0.25
        axs[0,0].plot(Ig, Qg, '.', label='g', color=blue, alpha=a, markersize=m)
        axs[0,0].plot(Ie, Qe, '.', label='e', color=red, alpha=a, markersize=m)
        
        if plot_f: axs[0,0].plot(If, Qf,'.', label='f', color='g', alpha=a, markersize=m)
        axs[0,0].plot(xg, yg, color='k', marker='o')
        axs[0,0].plot(xe, ye, color='k', marker='o')
        if plot_f: axs[0,0].plot(xf, yf, color='k', marker='o')

        # axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated')
        axs[0,0].axis('equal')
    else:
        fig=None

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((yf-yg),(xf-xg))

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [(xg+xe)/2-span, (xg+xe)/2+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[0,1].plot(Ig_new, Qg_new,'.', label='g', color=blue, alpha=a, markersize=m)
        axs[0,1].plot(Ie_new, Qe_new, '.', label='e', color=red, alpha=a, markersize=m)
        if plot_f: axs[0, 1].plot(If_new, Qf_new, '.', label='f', color='g', alpha=a, markersize=m)
        axs[0,1].plot(xg, yg, color='k', marker='o')
        axs[0,1].plot(xe, ye, color='k', marker='o')    
        if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

        # axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated')
        axs[0,1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color=blue, label='g', alpha=0.5)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color=red, label='e', alpha=0.5)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5)
        axs[1,0].set_ylabel('Counts')
        axs[1,0].set_xlabel('I [ADC levels]')       
        axs[1,0].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
        
    if plot: 
        axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts')
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), color=blue, label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), color=red, label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        axs[1,1].plot(np.nan, np.nan, color = 'white', label='Threshold: {:.2f}'.format(thresholds[0]))
        axs[1,1].plot(np.nan, np.nan, color = 'white', label='Angle: {:.2f}$^\circ$'.format(theta*180/np.pi))


        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]')
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi, fig # fids: ge, gf, ef

# ====================================================== #

class HistogramProgram(AveragerProgram):
    def __init__(self, soccfg, cfg):

        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)


    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        self.qubits = self.cfg.expt.qubits
        qTest = self.qubits[0]
        # self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch 

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.f_ge = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        if self.cfg.expt.pulse_f: 
            self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)

        self.f_ef = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        elif self.res_ch_types[qTest] == 'mux4':
            assert self.res_chs[qTest] == 6
            mask = [0,1,2,3] # indices of mux_freqs, mux_gains list to play

            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit_chan] = cfg.device.readout.frequency[qTest]
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit_chan] = cfg.device.readout.gain[qTest]
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
            
            ro_ch=self.adc_chs[qTest]
        else:
            ro_ch = self.adc_chs[qTest]

        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        self.declare_gen(ch=self.qubit_chs[qTest], nqz=cfg.hw.soc.dacs.qubit.nyquist[qTest])

        self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain[qTest]
        if self.cfg.expt.pulse_f:
            self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_chs)
            self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain
        
        # add qubit and readout pulses to respective channels
        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f and cfg.device.qubit.pulses.pi_ge.type == 'gauss':
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
        if self.cfg.expt.pulse_f and cfg.device.qubit.pulses.pi_ef.type== 'gauss':
            self.add_gauss(ch=self.qubit_ch, name="pi_ef_qubit", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)

        if self.res_ch_types[qTest] == 'mux4':
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        else: 
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], 
                                     gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest], 
                                     phase=self.deg2reg(-self.cfg.device.readout.phase[qTest], gen_ch = self.res_chs[qTest]))
     
        # print(-self.cfg.device.readout.phase)

        self.sync_all(200)
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = cfg.expt.qubits[0]

        if self.res_ch_types[qTest] == 'mux4':
            assert self.res_chs[qTest] == 6
            mask = [0,1,2,3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = cfg.device.readout.frequency[0:4]
            mux_gains = cfg.device.readout.gain[0:4]
            ro_ch=self.adc_chs[qTest]
        else:
            ro_ch = self.adc_chs[qTest]

        # Phase reset all channels
        #print("using phase reset")
        # for ch in self.gen_chs.keys():
        #     if not self.res_ch_types[qTest] == 'mux4':#self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels # is None or ch in self.res_chs:
        #        self.setup_and_pulse(ch=ch, style='const', freq=self.f_res_reg[qTest], phase=0, gain=0, length=self.us2cycles(0.1), phrst=1)
            #self.sync_all()

        if self.res_ch_types[qTest] == 'mux4':
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        else:
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(0, gen_ch = self.res_chs[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])  

        self.sync_all(100)

        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ge.type == 'gauss':
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge[qTest], phase=0, gain=self.pi_gain, phrst = 0,  waveform="pi_qubit")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.f_ge[qTest], phase=0, gain=self.pi_gain, phrst= 0, length=self.pi_sigma)
        self.sync_all()

        if self.cfg.expt.pulse_f:
            if cfg.device.qubit.pulses.pi_ef.type == 'gauss':
                self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef[qTest], phase=0, gain=self.pi_ef_gain, waveform="pi_ef_qubit")
            else: # const pulse
                self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_ef[qTest], phase=0, gain=self.pi_ef_gain, length=self.pi_ef_sigma)
        self.sync_all()

        self.measure(pulse_ch=self.res_chs[qTest], 
             adcs=[self.adc_chs[qTest]],
             adc_trig_offset=cfg.device.readout.trig_offset[qTest],
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]), #, gen_ch=self.res_ch
             )

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        cfg=AttrDict(self.cfg)
        qTest = cfg.expt.qubits[0]
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]
        return shots_i0, shots_q0


class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})                     

        data=dict()

        # Ground state shots
        cfg2 =deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress);
        data['Ig'], data['Qg'] = histpro.collect_shots()

        # Excited state shots
        if 'check_e' not in self.cfg.expt:
            self.check_e = True
        else: self.check_e = self.cfg.expt.check_e
        if self.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, )
            data['Ie'], data['Qe'] = histpro.collect_shots()

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True 
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,progress=progress, )
            data['If'], data['Qf'] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data=self.data
        
        fids, thresholds, angle, _ = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        
        return data

    def display(self, data=None, span=None, verbose=True, plot_e=True, plot_f=False, **kwargs):
        if data is None:
            data=self.data 
        
        fids, thresholds, angle, fig = hist(data=data, plot=True, verbose=verbose, span=span)
            
        print(f'ge fidelity (%): {100*fids[0]}')
        if 'expt' not in self.cfg: 
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f'gf fidelity (%): {100*fids[1]}')
            print(f'ef fidelity (%): {100*fids[2]}')
        print(f'rotation angle (deg): {angle}')
        print(f'threshold ge: {thresholds[0]}')
        if self.cfg.expt.check_f:
            print(f'threshold gf: {thresholds[1]}')
            print(f'threshold ef: {thresholds[2]}')
        qubit = self.cfg.expt.qubits[0]
        imname = self.fname.split("\\")[-1]
        fig.suptitle(f'Single Shot Histogram Analysis Q{qubit}')
        fig.tight_layout()
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)



# ====================================================== #
class SingleShotOptExperiment(Experiment):
    """
    Single Shot optimization experiment over readout parameters
    expt = dict(
        reps: number of shots per expt
        start_f: start frequency (MHz)
        step_f: frequency step (MHz)
        expts_f: number of experiments in frequency

        start_gain: start gain (dac units)
        step_gain: gain step (dac units)
        expts_gain: number of experiments in gain sweep

        start_len: start readout len (dac units)
        step_len: length step (dac units)
        expts_len: number of experiments in length sweep

        check_f: optimize fidelity for g/f (as opposed to g/e)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Histogram', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=True):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"]*np.arange(self.cfg.expt["expts_len"])
        # print(fpts)
        # print(gainpts)
# print(lenpts)
        if 'save_data' not in self.cfg.expt: 
            self.cfg.expt.save_data = False
        
        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        if 'check_f' not in self.cfg.expt: 
            check_f = False
        else:
            check_f = self.cfg.expt.check_f
        qubit = self.cfg.expt.qubit
        Ig, Ie, Qg, Qe = [], [], [], []
        if check_f: If, Qf = [], []
        for f_ind, f in enumerate(tqdm(fpts, disable=not progress)):
            Ig.append([]); Ie.append([]); Qg.append([]); Qe.append([])
            if check_f: If.append([]); Qf.append([])
            for g_ind, gain in enumerate(gainpts):
                Ig[-1].append([]); Ie[-1].append([]); Qg[-1].append([]); Qe[-1].append([])
                if check_f: If[-1].append([]); Qf[-1].append([])
                for l_ind, l in enumerate(lenpts):
                    shot = HistogramExperiment(soccfg=self.soccfg, config_file=self.config_file, im=self.im)
                    shot.cfg = self.cfg
                    shot.cfg.device.readout.frequency = float(f)
                    if int_rgain: 
                        shot.cfg.device.readout.gain = int(gain)
                    shot.cfg.device.readout.readout_length = float(l) 
                    check_e = True
                    
                    shot.cfg.expt = dict(reps=self.cfg.expt.reps, check_e=check_e, check_f=check_f, qubit=self.cfg.expt.qubit, save_data=self.cfg.expt.save_data, qubits=self.cfg.expt.qubits, qubit_chan = self.cfg.expt.qubit_chan)
                    shot.go(analyze=False, display=False, progress=progress, save=False);
                    Ig[-1][-1].append(shot.data['Ig']); Ie[-1][-1].append(shot.data['Ie']); 
                    Qg[-1][-1].append(shot.data['Qg']); Qe[-1][-1].append(shot.data['Qe'])
                    if check_f: If[-1][-1].append(shot.data['If']); Qf[-1][-1].append(shot.data['Qf'])
                    results = shot.analyze(verbose=False)
                    fid[f_ind, g_ind, l_ind] = results['fids'][0] if not check_f else results['fids'][1]
                    threshold[f_ind, g_ind, l_ind] = results['thresholds'][0] if not check_f else results['thresholds'][1]
                    angle[f_ind, g_ind, l_ind] = results['angle']
                    # print(f'freq: {f}, gain: {gain}, len: {l}')
                    # print(f'\tfid ge [%]: {100*results["fids"][0]}')
                    if check_f: print(f'\tfid gf [%]: {100*results["fids"][1]}')


        if self.cfg.expt.save_data: 
            self.data = dict(fpts=fpts, gainpts=gainpts, lenpts=lenpts, fid=fid, threshold=threshold, angle=angle, Ig=Ig, Ie=Ie, Qg=Qg, Qe=Qe)
            if check_f: self.data['If'] = If; self.data['Qf'] = Qf
        else:
            self.data = dict(fpts=fpts, gainpts=gainpts, lenpts=lenpts, fid=fid, threshold=threshold, angle=angle)
        
        return self.data

    def analyze(self, data=None, **kwargs):
        if data == None: data = self.data
        fid = data['fid']
        threshold = data['threshold']
        angle = data['angle']
        fpts = data['fpts']
        gainpts = data['gainpts']
        lenpts = data['lenpts']

        imax = np.unravel_index(np.argmax(fid), shape=fid.shape)
        print(imax)
        print(fpts)
        print(gainpts)
        print(lenpts)
        print(f'Max fidelity {100*fid[imax]} %')
        print(f'Set params: \n angle (deg) {-angle[imax]} \n threshold {threshold[imax]} \n freq [Mhz] {fpts[imax[0]]} \n Gain [DAC units] {gainpts[imax[1]]} \n readout length [us] {lenpts[imax[2]]}')
        self.data['freq'] = fpts[imax[0]]
        self.data['gain'] = gainpts[imax[1]]
        self.data['length'] = lenpts[imax[2]]

        return imax

    def display(self, data=None, **kwargs):
        if data is None:
            data=self.data 
        
        fid = data['fid']
    
        fpts = data['fpts'] # outer sweep, index 0
        gainpts = data['gainpts'] # middle sweep, index 1
        lenpts = data['lenpts'] # inner sweep, index 2
        # if len(fpts)>1 and gainpts>1 and lenpts>1: 
        #     sweep3d = True
        # elif fpts>1 and gainpts>1 or fpts>1 and lenpts>1 or gainpts>1 and lenpts>1:
        #     sweep2d = True
        #     sweep3d = False
        # else:
        #     sweep2d= False
        #     sweep3d = False
        fig = plt.figure(figsize=(9,5.5))
        if len(fpts)>1: 
            xval = fpts
            xlabel='Frequency [MHz]'
            var1 = gainpts
            var2 = lenpts
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(xval, 100*fid[:,v1_ind, v2_ind], 'o-', label=f'{v1:1.0f}, {v2:.2f}')
        elif len(gainpts)>1:
            xval = gainpts
            xlabel='Gain [DAC units]'
            var1 = fpts
            var2 = lenpts
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(xval, 100*fid[v1_ind,:, v2_ind], 'o-', label=f'{v1:.2f}, {v2:.2f}')
        else:
            xval = lenpts
            xlabel='Readout length [us]'
            var1 = fpts
            var2 = gainpts
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(xval, 100*fid[v1_ind, v2_ind,:], 'o-', label=f'{v2:1.0f},  {v1:.2f}')
            
        plt.xlabel(xlabel)
        plt.ylabel(f'Fidelity [%]')
        plt.legend()
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname