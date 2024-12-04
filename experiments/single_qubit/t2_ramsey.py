import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from datetime import datetime

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class RamseyProgram(RAveragerProgram):
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
        self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF
        self.acStark = self.cfg.expt.acStark    

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        
        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
        else: qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        if self.checkZZ:
            if qTest == 1: 
                self.f_Q1_ZZ_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q1_ZZ]
            else: 
                self.f_Q_ZZ1_reg = [self.freq2reg(f, gen_ch=self.qubit_chs[qTest]) for f in cfg.device.qubit.f_Q_ZZ1]
        
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
        elif self.res_ch_types[qTest] == 'mux4':
            assert self.res_chs[qTest] == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit_chan] = cfg.device.readout.frequency[qTest]
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit_chan] = cfg.device.readout.gain[qTest]
            ro_ch=self.adc_chs[qTest]
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # declare registers for phase incrementing
        self.r_wait = 3
        self.r_phase2 = 4
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_phase = self.sreg(self.qubit_chs[qTest], "freq")
            self.r_phase3 = 5 # for storing the left shifted value
        else: self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        # define pi2sigma as the pulse that we are calibrating with ramsey
        self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest]/2, gen_ch=self.qubit_chs[qTest])
        self.f_pi_test_reg = self.f_ge_reg[qTest] # freq we are trying to calibrate
        self.gain_pi_test = self.cfg.device.qubit.pulses.pi_ge.gain[qTest] # gain of the pulse we are trying to calibrate
        
        if self.checkZZ:
            self.pisigma_ge_qZZ = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qZZ], gen_ch=self.qubit_chs[qZZ])
            if qTest == 1:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_Q1_ZZ.sigma[qSort]/2, gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q1_ZZ_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_pi_test = self.cfg.device.qubit.pulses.pi_Q1_ZZ.gain[qSort] # gain of the pulse we are trying to calibrate
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q1_ZZ_reg[qZZ] # freq we are trying to calibrate
            else:
                self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort], gen_ch=self.qubit_chs[qTest])
                self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_Q_ZZ1.sigma[qSort]/2, gen_ch=self.qubit_chs[qTest])
                self.f_ge_init_reg = self.f_Q_ZZ1_reg[qSort] # freq to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_ge_init = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[qSort] # gain to use if wanting to doing ge for the purpose of doing an ef pulse
                self.gain_pi_test = self.cfg.device.qubit.pulses.pi_Q_ZZ1.gain[qSort] # gain of the pulse we are trying to calibrate
                if 'f_pi_test' not in self.cfg.expt: self.f_pi_test_reg = self.f_Q_ZZ1_reg[qSort] # freq we are trying to calibrate
        
        if self.checkEF:
            self.pi2sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qTest]/2, gen_ch=self.qubit_chs[qTest])
            self.f_pi_test_reg = self.f_ef_reg[qTest] # freq we are trying to calibrate
            self.gain_pi_test = self.cfg.device.qubit.pulses.pi_ef.gain[qTest] # gain of the pulse we are trying to calibrate
        if self.acStark: 
            self.stark_freq = self.freq2reg(cfg.expt.stark_freq, gen_ch=self.qubit_chs[qTest])
            self.stark_gain = self.cfg.expt.stark_gain # gain of the pulse we are trying to calibrate
        #print(self.f_pi_test_reg)
        #print(self.f_ge_reg[qTest])
        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test", sigma=self.pi2sigma, length=self.pi2sigma*4)
        if self.checkZZ:
            self.add_gauss(ch=self.qubit_chs[qZZ], name="pi_qubitZZ", sigma=self.pisigma_ge_qZZ, length=self.pisigma_ge_qZZ*4)
        if self.checkEF:
            self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)

        # add readout pulses to respective channels
        if self.res_ch_types[qTest] == 'mux4':
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        else: 
            self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest], phase=self.deg2reg(-self.cfg.device.readout.phase[qTest], gen_ch = self.res_chs[qTest]))

        # initialize wait registers
        self.safe_regwi(self.q_rps[qTest], self.r_wait, self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.q_rps[qTest], self.r_phase2, 0) 

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        if self.checkZZ: qZZ, qTest = self.qubits
        else: qTest = self.qubits[0]

        # Phase reset all channels
        # for ch in self.gen_chs.keys():
        #     if self.gen_chs[ch]['mux_freqs'] is None: # doesn't work for the mux channels
        #         # print('resetting', ch)
        #         self.setup_and_pulse(ch=ch, style='const', freq=100, phase=0, gain=100, length=10, phrst=1)
        #     self.sync_all()
        # self.sync_all(10)

        # initializations as necessary
        if self.checkZZ:
            self.setup_and_pulse(ch=self.qubit_chs[qZZ], style="arb", phase=0, freq=self.f_ge_reg[qZZ], gain=cfg.device.qubit.pulses.pi_ge.gain[qZZ], waveform="pi_qubitZZ")
            self.sync_all(5)
        if self.checkEF:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")
            self.sync_all(5)

        # play pi/2 pulse with the freq that we want to calibrate
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_pi_test_reg, phase=0, gain=self.gain_pi_test, waveform="pi2_test")

        # wait advanced wait time
        self.sync_all()

            #self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.stark_freq, phase=0, gain=self.stark_gain, waveform="stark_test")
        self.sync(self.q_rps[qTest], self.r_wait)

       
        # play pi/2 pulse with advanced phase (all regs except phase are already set by previous pulse)
        if self.qubit_ch_types[qTest] == 'int4':
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rps[qTest], self.r_phase3, self.r_phase3, '|', self.f_pi_test_reg)
            self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase3, "+", 0)        
        #self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_pi_test_reg, gain=self.gain_pi_test, waveform="pi2_test")
        #self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_pi_test_reg, phase=0, gain=self.gain_pi_test, waveform="pi2_test")
        else: self.mathi(self.q_rps[qTest], self.r_phase, self.r_phase2, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])

        if self.checkEF: # map excited back to qubit ground state for measurement
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")

        # align channels and measure
        self.sync_all(5)
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )

    def update(self):
        if self.checkZZ: qZZ, qTest = self.qubits
        else: qTest = self.qubits[0]

        phase_step = self.deg2reg(360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step, gen_ch=self.qubit_chs[qTest]) # phase step [deg] = 360 * f_Ramsey [MHz] * tau_step [us]
        self.mathi(self.q_rps[qTest], self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update the time between two π/2 pulses
        self.mathi(self.q_rps[qTest], self.r_phase2, self.r_phase2, '+', phase_step) # advance the phase of the LO for the second π/2 pulse


class RamseyExperiment(Experiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='Ramsey', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
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
        
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        ramsey = RamseyProgram(soccfg=self.soccfg, cfg=self.cfg)
        #print(ramsey)
        
        xpts, avgi, avgq = ramsey.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)        
 
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase

        current_time = current_time.encode('ascii','replace')
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases, 'time':current_time} 
        self.data=data
        return data

    def analyze(self, data=None, fit=True, fit_twofreq=False,debug=False, **kwargs):
        if data is None:
            data=self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # fitparams=[yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01] # two fit freqs
            # Remove the first and last point from fit in case weird edge measurements
            fitparams = None
            if fit_twofreq: fitfunc = fitter.fittwofreq_decaysin
            else: fitfunc = fitter.fitdecaysin


            ydata_lab = ['amps', 'avgi', 'avgq']
            for i, ydata in enumerate(ydata_lab):
                data['fit_' + ydata], data['fit_err_' + ydata], data['init_guess_'+ydata] = fitfunc(data['xpts'], data[ydata], fitparams=fitparams, debug=debug)
                if isinstance(data['fit_'+ydata], (list, np.ndarray)): 
                    data['f_adjust_ramsey_'+ydata] = sorted((self.cfg.expt.ramsey_freq - data['fit_'+ydata][1], -self.cfg.expt.ramsey_freq - data['fit_'+ydata][1]), key=abs)              

                if fit_twofreq:
                    data['f_adjust_ramsey_'+ydata+'2'] = sorted((self.cfg.expt.ramsey_freq - data['fit_' + ydata][7], -self.cfg.expt.ramsey_freq - data['fit_' + ydata][6]), key=abs)

            fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(self.data, get_best_data_params=['f_adjust_ramsey'])
            data['t2r_adjust']=t2r_adjust
            r2 = fitter.get_r2(data['xpts'], data[i_best], fitter.decaysin, fit_pars)
            print('R2:', r2)
            data['r2']=r2

            data['fit_err']=np.mean(np.abs(fit_err/fit_pars))
            print('fit_err:', data['fit_err'])

            data['best_fit'] = fit_pars
            i_best = i_best.encode("ascii", "ignore")
            data['i_best']=i_best
            print(f'Best fit: {i_best}')

            if self.cfg.expt.checkEF: 
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubits[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubits[0]]
            
            if t2r_adjust[0] < np.abs(t2r_adjust[1]):
                new_freq = f_pi_test + t2r_adjust[0]
            else:       
                new_freq = f_pi_test + t2r_adjust[1]
            data['new_freq']=new_freq
            
        return data

    def display(self, data=None, fit=True, fit_twofreq=False,debug=False,plot_all=False, **kwargs):
        if data is None:
            data=self.data

        self.qubits = self.cfg.expt.qubits
        self.checkZZ = self.cfg.expt.checkZZ
        self.checkEF = self.cfg.expt.checkEF

        if self.checkZZ: # [x, 1] means test Q1 with ZZ from Qx; [1, x] means test Qx with ZZ from Q1, sort by Qx in both cases
            assert len(self.qubits) == 2
            assert 1 in self.qubits
            qZZ, qTest = self.qubits
            qSort = qZZ # qubit by which to index for parameters on qTest
            if qZZ == 1: qSort = qTest
        else: qTest = self.qubits[0]

        f_pi_test = self.cfg.device.qubit.f_ge[qTest]
        if self.checkZZ:
            if qTest == 1: f_pi_test = self.cfg.device.qubit.f_Q1_ZZ[qSort]
            else: f_pi_test = self.cfg.device.qubit.f_Q_ZZ1[qSort]
        if self.checkEF: f_pi_test = self.cfg.device.qubit.f_ef[qTest]

        title = ('EF' if self.checkEF else '') + f'Ramsey on Q{qTest}' + (f'with Q{qZZ} in e' if self.checkZZ else '') 

        if fit_twofreq: fitfunc = fitter.twofreq_decaysin
        else: fitfunc = fitter.decaysin
        print(f'Current pi pulse frequency: {f_pi_test}')
        title = title + f' (Ramsey Freq: {self.cfg.expt.ramsey_freq:.3f} MHz)'        

        if plot_all:
            fig, ax=plt.subplots(3, 1, figsize=(9, 11))
            fig.suptitle(title)
            ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
            ydata_lab = ['amps', 'avgi', 'avgq']
        else:
            fig, a=plt.subplots(1, 1, figsize=(7.5, 4))
            a.set_title(title)
            ylabels = ["I [ADC units]"]
            ydata_lab = ['avgi']
            ax = [a]
        
        xlabel = "Wait Time (us)"
        fitfunc=fitter.decaysin
        for i, ydata in enumerate(ydata_lab):
            ax[i].plot(data["xpts"], data[ydata],'o-')
        
            if fit:
                p = data['fit_'+ydata]
                pCov = data['fit_err_amps']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3} \n'
                captionStr += f'Frequency [MHz]: {p[1]:.3} $\pm$ {np.sqrt(pCov[1][1]):.3}'
                ax[i].plot(data["xpts"], fitfunc(data["xpts"], *p), label=captionStr)

                # Plot the decaying exponential
                x0 = -(p[2]+180)/360/p[1]
                ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], p[0], x0, p[3]), color='0.2', linestyle='--')
                ax[i].plot(data["xpts"], fitter.expfunc2(data['xpts'], p[4], -p[0], x0, p[3]), color='0.2', linestyle='--')

                ax[i].set_ylabel(ylabels[i])
                ax[i].set_xlabel(xlabel)
                ax[i].legend(loc='upper right')
                if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
         
            if debug: 
                pinit = data['init_guess_'+ydata]
                print(pinit)
                plt.plot(data["xpts"], fitfunc(data["xpts"], *pinit), label='Initial Guess')
    
            if fit_twofreq:
                print('Beating frequency from fit [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi2"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi2"][1]}')
       
        imname = self.fname.split("\\")[-1]
        fig.tight_layout()
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

        print('New pi pulse frequency {:3.4f}:\n'.format(data['new_freq']))
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname