import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from datetime import datetime

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

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


class RamseyEchoExperiment(Experiment):
    """
    Ramsey Echo Experiment
    Experimental Config:
    expt = dict(
        start: total wait time b/w the two pi/2 pulses start sweep [us]
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

    def __init__(self, soccfg=None, path='', prefix='RamseyEcho', config_file=None, progress=None, im=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, im=im)

    def acquire(self, progress=False, debug=False):
        assert self.cfg.expt.cp != self.cfg.expt.cpmg, 'Must select either CP or CPMG experiment!'

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

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        prog = RamseyEchoProgram(soccfg=self.soccfg, cfg=self.cfg)
        xpts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        current_time = current_time.encode('ascii','replace')
        data={'xpts': xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases, 'time':current_time}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, debug=False, **kwargs):
        if data is None:
            data=self.data
        if fit:
            fitparams = None
            
            fitfunc = fitter.fitdecaysin

            ydata_lab = ['amps', 'avgi', 'avgq']
            for i, ydata in enumerate(ydata_lab):
                data['fit_' + ydata], data['fit_err_' + ydata], data['init_guess_'+ydata] = fitfunc(data['xpts'], data[ydata], fitparams=fitparams, debug=debug)
                if isinstance(data['fit_'+ydata], (list, np.ndarray)): 
                    data['f_adjust_ramsey_'+ydata] = sorted((self.cfg.expt.ramsey_freq - data['fit_'+ydata][1], -self.cfg.expt.ramsey_freq - data['fit_'+ydata][1]), key=abs)              
                        
            fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(self.data, get_best_data_params=['f_adjust_ramsey'])

            r2 = fitter.get_r2(data['xpts'], data[i_best], fitter.decaysin, fit_pars)
            print('R2:', r2)
            data['r2']=r2

            data['fit_err']=np.mean(np.abs(fit_err/fit_pars))
            print('fit_err:', data['fit_err'])
            data['best_fit'] = data['fit_'+i_best]

            i_best = i_best.encode("ascii", "ignore")
            data['i_best']=fit_pars
            
            print(f'Best fit: {i_best}')
            f_pi_test = self.cfg.device.qubit.f_ge

            if t2r_adjust[0] < np.abs(t2r_adjust[1]):
                new_freq = f_pi_test + t2r_adjust[0]
            else:       
                new_freq = f_pi_test + t2r_adjust[1]
            data['new_freq']=new_freq

        return data

    def display(self, data=None, fit=True, debug=False, **kwargs):
        if data is None:
            data=self.data
        qubit = self.cfg.expt.qubit
        title=f"Ramsey Echo Q{qubit} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",

        print(f'Current qubit frequency: {self.cfg.device.qubit.f_ge}')
        fitfunc=fitter.decaysin

        fig, ax=plt.subplots(3, 1, figsize=(9, 11))
        xlabel = "Wait Time (us)"
        ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
        fig.suptitle(title)
        ydata_lab = ['amps', 'avgi', 'avgq']
        fitfunc=fitter.decaysin
        for i, ydata in enumerate(ydata_lab):
            ax[i].plot(data["xpts"], data[ydata],'o-')
        
            if fit:
                p = data['fit_'+ydata]
                pCov = data['fit_err_amps']
                captionStr = f'$T_2$ Echo fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3} \n'
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

            #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            # if p[1] > 2*self.cfg.expt.ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the qubit frequency!')
            
        plt.show()
        imname = self.fname.split("\\")[-1]
        fig.tight_layout()
        fig.savefig(self.fname[0:-len(imname)]+'images\\'+imname[0:-3]+'.png')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname