import matplotlib.pyplot as plt
import numpy as np
from qick import *
import copy
import seaborn as sns
from exp_handling.datamanagement import AttrDict
from gen.qick_experiment import QickExperiment
from gen.qick_program import QickProgram


import slab_qick_calib.config as config
blue = "#4053d3"
red = "#b51d14"
int_rgain = True
import slab_qick_calib.calib.readout_helpers as helpers


# ====================================================== #

class HistogramProgram(QickProgram):

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        if cfg.expt.active_reset:
            self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        else:
            self.phase = 0
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="")

        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        if cfg.expt.pulse_f:
            super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")
        self.delay(0.5) # give the tProc some time for initial setup        

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], ddr4=True,pins=[0],t=self.trig_offset)

        if cfg.expt.active_reset:
            self.reset(7)


    def reset(self, i):
        super().reset(i)
        
    
    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            #nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]# / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1] #/ nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots


class HistogramExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):

        if prefix is None:
            prefix = f"single_shot_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            soft_avgs=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset = False,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
            ddr4 = False,
        )
        
        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()
        
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if 'setup_reset' in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots
        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        histpro = HistogramProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
        #histpro.config_all(self.im[self.cfg.aliases.soc])

        if self.cfg.expt.ddr4:
            n_transfers = 1500000 # each transfer (aka burst) is 256 decimated samplesn_transfers = 100000 # each transfer (aka burst) is 256 decimated samples
            nt = n_transfers
            # Arm the buffers
            #self.soc.arm_ddr4(ch=self.cfg.expt.qubit_chan, nt=n_transfers)
            #cfg.hw.soc.adcs.readout.ch[q]
            self.im[self.cfg.aliases.soc].arm_ddr4(ch=self.cfg.expt.qubit_chan, nt=n_transfers)

        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        

        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]
        if self.cfg.expt.active_reset:
            data["Igr"]=iq_list[0][1:,:, 0]
        
        if self.cfg.expt.ddr4:
            iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
            #iq_ddr4 = self.soc.get_ddr4(100)
            t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
            data['t_g'] = t
            data['iq_ddr4_g'] = iq_ddr4
        irawg, qrawg = histpro.collect_shots()
        
        rawd = [irawg[-1], qrawg[-1]]
        #print("buffered readout:", rawd)

        # Excited state shots
        if self.cfg.expt.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            if self.cfg.expt.ddr4:
                self.im[self.cfg.aliases.soc].arm_ddr4(ch=self.cfg.expt.qubit_chan, nt=n_transfers)
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )
            if self.cfg.expt.ddr4:
                iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
                t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
                data['t_e'] = t
                data['iq_ddr4_e'] = iq_ddr4

            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]
            irawe, qraw = histpro.collect_shots()
            #rawd = [iraw[-1], qraw[-1]]
            #print("buffered readout:", rawd)
            #print("feedback readout:", self.soc.read_mem(2,'dmem'))
            if self.cfg.expt.active_reset:
                data["Ier"]=iq_list[0][1:,:, 0]
            #print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )
            data["If"], data["Qf"] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

        params, _ = helpers.hist(
            data=data, plot=False, span=span, verbose=verbose
        )
        data.update(params)
        try:
            data2, p, paramsg, paramse2 = helpers.fit_single_shot(data, plot=False)
            data.update(p)
            data["vhg"]=data2["vhg"]
            data["histg"]=data2["histg"]
            data["vhe"]=data2["vhe"]
            data["histe"]=data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
        except:
            print('Fits failed')
             
        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        params, fig = helpers.hist(
            data=data, plot=plot, verbose=verbose, span=span, ax=ax, qubit=self.cfg.expt.qubit[0]
        )
        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if verbose:        
            print(f"ge Fidelity (%): {100*fids[0]:.3f}")

            if self.cfg.expt.check_f:
                print(f"gf Fidelity (%): {100*fids[1]:.3f}")
                print(f"ef Fidelity (%): {100*fids[2]:.3f}")
            print(f"Rotation angle (deg): {angle:.3f}")
            print(f"Threshold ge: {thresholds[0]:.3f}")
            if self.cfg.expt.check_f:
                print(f"Threshold gf: {thresholds[1]:.3f}")
                print(f"Threshold ef: {thresholds[2]:.3f}")
        imname = self.fname.split("\\")[-1]

        if savefig:
            plt.show()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

    def update(self, cfg_file, freq=True, fast=False, verbose=True):
        qi = self.cfg.expt.qubit[0]
        
        config.update_readout(cfg_file, 'phase', self.data['angle'], qi, verbose=verbose)        
        config.update_readout(cfg_file, 'threshold', self.data['thresholds'][0], qi, verbose=verbose)
        config.update_readout(cfg_file, 'fidelity', self.data['fids'][0], qi, verbose=verbose)
        if not fast:
            config.update_readout(cfg_file, 'sigma', self.data['sigma'], qi, verbose=verbose)
            config.update_readout(cfg_file, 'tm', self.data['tm'], qi, verbose=verbose)
            if self.data['fids'][0]>0.07:
                config.update_qubit(cfg_file, 'tuned_up', True, qi, verbose=verbose)
            else:
                config.update_qubit(cfg_file, 'tuned_up', False, qi, verbose=verbose)
                print('Readout not tuned up')

    def check_reset(self): 
        nbins=75
        fig, ax = plt.subplots(2,1, figsize=(6,7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = helpers.make_hist(self.data['Ig'], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b  = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data['Igr']))
        ve, histe = helpers.make_hist(self.data['Ie'], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data['Igr'])):
            v, hist = helpers.make_hist(self.data['Igr'][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')
            v, hist = helpers.make_hist(self.data['Ier'][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f'{i+1}')

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind= find_bin_closest_to_value(v, self.data['ie'])
        ind_e= find_bin_closest_to_value(ve, self.data['ie'])
        ind_g= find_bin_closest_to_value(vg, self.data['ie'])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g")

        self.data['reset_e'] = reset_level/e_level
        self.data['reset_g'] = reset_level/g_level



        

        
        ax[0].legend()

        ax[0].set_title('Ground state')
        ax[1].set_title('Excited state')
        plt.show()


# ====================================================== #
