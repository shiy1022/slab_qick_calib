class Scan: 

    def __init__(self, cfg_dict, qi, params, min_r2=0.1, max_err=0.5):
        self.cfg_dict = cfg_dict
        self.qi = qi
        self.min_r2 = min_r2
        self.max_err = max_err

    def make_scan(self, scan_name, params={}):
        pass 

    def run(self, scan_name):

        prog = self.make_scan(cfg_dict['soc'], cfg_dict['expt_path'], cfg_dict['cfg_file'], qi, cfg_dict['im'], **params)
        if 'fit_err' in prog.data and 'r2' in prog.data and prog.data['fit_err'] < self.max_err and prog.data['r2'] >self. min_r2:
            return True, prog
        elif 'fit_err' not in prog.data or 'r2' not in prog.data:
            return prog
        else:
            print('Fit failed')
            return False, prog 