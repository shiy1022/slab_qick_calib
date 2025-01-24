import numpy as np
import scipy as sp
import cmath
import traceback

# ====================================================== #

"""
Compare the fit between the check_measures (amps, avgi, and avgq by default) in data, using the compare_param_i-th parameter to do the comparison. Pick the best method of measurement out of the check_measures, and return the fit, fit_err, and any other get_best_data_params corresponding to that measurement.

If fitfunc is specified, uses R^2 to determine best fit.
"""
def get_best_fit(data, fitfunc=None, prefixes=['fit'], check_measures=('amps', 'avgi', 'avgq'), get_best_data_params=(), override=None):
    fit_errs = [data[f'{prefix}_err_{check}'] for check in check_measures for prefix in prefixes]

    # fix the error matrix so "0" error is adjusted to inf
    for fit_err_check in fit_errs:
        for i, fit_err in enumerate(np.diag(fit_err_check)):
            if fit_err == 0: fit_err_check[i][i] = np.inf

    fits = [data[f'{prefix}_{check}'] for check in check_measures for prefix in prefixes]
    if override is not None and override in check_measures:
        i_best = np.argwhere(np.array(check_measures) == override)[0][0]
        print(i_best)
    else:
        if fitfunc is not None:
            ydata = [data[check] for check in check_measures]  # need to figure out how to make this support multiple qubits readout
            xdata = data['xpts']

            # residual sum of squares
            ss_res_checks = np.array([np.sum((fitfunc(xdata, *fit_check) - ydata_check)**2) for fit_check, ydata_check in zip(fits, ydata)])
            # total sum of squares
            ss_tot_checks = np.array([np.sum((np.mean(ydata_check) - ydata_check)**2) for ydata_check in ydata])
            # R^2 value
            r2 = 1- ss_res_checks / ss_tot_checks
            par_error_norm=[]
            for fit, fit_err_check in zip(fits, fit_errs):
                par_error_norm.append(np.mean(np.sqrt(np.abs(np.diag(fit_err_check)))/np.abs(fit)))

            par_error_norm=np.abs(par_error_norm)
            # override r2 value if fit is bad
            for icheck, fit_err_check in enumerate(fit_errs):
                for i, fit_err in enumerate(np.diag(fit_err_check)):
                    if fit_err == np.inf: r2[icheck] = np.inf
            i_best = np.argmin(par_error_norm)            
        else:
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            # i_best = np.argmin([np.sqrt(np.abs(fit_err[compare_param_i][compare_param_i] / fit[compare_param_i])) for fit, fit_err in zip(fits, fit_errs)])
            errs = [np.average(np.sqrt(np.abs(np.diag(fit_err))) / np.abs(fit)) for fit, fit_err in zip(fits, fit_errs)]
            # print([np.sqrt(np.abs(np.diag(fit_err))) / np.abs(fit) for fit, fit_err in zip(fits, fit_errs)])
            for i_err, err in enumerate(errs):
                if err == np.nan: errs[i_err] = np.inf
            # print(errs)
            i_best = np.argmin(errs)
            # print(i_best)

    best_data = [fits[i_best], fit_errs[i_best]]
    best_meas = check_measures[i_best]

    for param in get_best_data_params:
        best_data.append(data[f'{param}_{best_meas}'])
    best_data.append(best_meas)
    return best_data


def get_r2(xdata,ydata, fitfunc, fit_params): 
    
    ss_res = np.sum((fitfunc(xdata, *fit_params) - ydata)**2)
    # total sum of squares
    ss_tot = np.sum((np.mean(ydata) - ydata)**2)
    # R^2 value
    r2 = 1- ss_res / ss_tot
    return r2

def fix_phase(p): 
    if p[2] > 180: 
        p[2] = p[2] - 360
    elif p[2] < -180: 
        p[2] = p[2] + 360
    if p[2] < 0: 
        pi_gain = (1/2 - p[2]/180)/2/p[1]
    else: 
        pi_gain= (3/2 - p[2]/180)/2/p[1]
    return pi_gain
        
# ====================================================== #

def expfunc(x, *p):
    y0, yscale, decay = p
    return y0 + yscale*np.exp(-x/decay)

def expfunc2(x, *p):
    y0, yscale, x0, decay = p
    return y0 + yscale*np.exp(-(x-x0)/decay)

def fitexp(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*3
    if fitparams[0] is None: fitparams[0] = ydata[-1]
    if fitparams[1] is None: fitparams[1] = ydata[0]-ydata[-1]
    #if fitparams[2] is None: fitparams[2] = xdata[0]
    if fitparams[2] is None: fitparams[2] = (xdata[-1]-xdata[0])/4
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(expfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams

# ====================================================== #

def lorfunc(x, *p):
    y0, yscale, x0, xscale = p
    return y0 + yscale/(1+(x-x0)**2/xscale**2)

def fitlor(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*4
    if fitparams[0] is None: fitparams[0] = (ydata[0] + ydata[-1])/2
    if fitparams[1] is None: fitparams[1] = max(ydata) - min(ydata)
    if fitparams[2] is None: fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]
    if fitparams[3] is None: fitparams[3] = (max(xdata)-min(xdata))/10
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(lorfunc, xdata, ydata, p0=fitparams)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams

# ====================================================== #

def sinfunc(x, *p):
    yscale, freq, phase_deg, y0 = p
    return yscale*np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) + y0

def fitsin(xdata, ydata, fitparams=None, debug=False):
    if fitparams is None: fitparams = [None]*4
    
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    if fitparams[0] is None: fitparams[0]=1/2*(max(ydata)-min(ydata))
    if fitparams[1] is None: fitparams[1]=max_freq
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=np.mean(ydata)
    bounds = (
        [0.5*fitparams[0], 1e-3, -360, np.min(ydata)],
        [2*fitparams[0], 1000, 360, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(sinfunc, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams 
    

# ====================================================== #

def decaysin(x, *p):
    yscale, freq, phase_deg, decay, y0 = p
    x0 = -(phase_deg+180)/360/freq
    # x0 = 0
    return yscale * np.sin(2*np.pi*freq*x + phase_deg*np.pi/180) * np.exp(-(x-x0)/decay) + y0

def fitdecaysin(xdata, ydata, fitparams=None, debug=False):
    # yscale, freq, phase_deg, decay, y0
    
    if fitparams is None: fitparams = [None]*5
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    bounds = (
        [0.6*fitparams[0], 1e-3, -360, 0.1, np.min(ydata)],
        [1.5*fitparams[0], 1e3, 360, np.inf, np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams

    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        try: 
            fitparams[2]=-fitparams[2]
            pOpt, pCov = sp.optimize.curve_fit(decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        except:
            print('Warning: fit failed!')
            pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams



def decayslopesin(x, *p):
    yscale, freq, phase_deg, decay, y0, slope = p
    x0 = -(phase_deg+180)/360/freq
    # x0 = 0
    return yscale * (np.sin(2*np.pi*freq*x + phase_deg*np.pi/180)+slope) * np.exp(-(x-x0)/decay) + y0

def fitdecayslopesin(xdata, ydata, fitparams=None, debug=False):
    # yscale, freq, phase_deg, decay, y0
    
    if fitparams is None: fitparams = [None]*6
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata)
    if fitparams[5] is None: fitparams[5]=0
    bounds = (
        [0.6*fitparams[0], 1e-3, -360, 0.1, np.min(ydata), -np.inf],
        [1.5*fitparams[0], 1e3, 360, np.inf, np.max(ydata), np.inf]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams

    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        try: 
            fitparams[2]=-fitparams[2]
            pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams, bounds=bounds)
        except:
            print('Warning: fit failed!')
            pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams
    

# ====================================================== #

def fourier_init(xdata, ydata, debug):
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)

    sorted_fourier = np.sort(np.abs(fourier))
    max_ind = np.argwhere(np.abs(fourier) == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(np.abs(fourier) == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        fourier[np.argmax(np.abs(fourier))] = np.nan
        plt.plot(fft_freqs, fourier,'.')
        plt.xlim(left=0)
        #plt.plot(fourier, fft_phases,'.')

        print('Max phase is ' + str(max_phase))
        print('Max freq is ' + str(max_freq))

    return max_freq, max_phase 

def twofreq_decaysin(x, *p):
    yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01 = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1)

def fittwofreq_decaysin(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*10
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1]-xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]
    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]
    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]
    if fitparams[0] is None: fitparams[0]=max(ydata)-min(ydata)
    if fitparams[1] is None: fitparams[1]=max_freq
    # if fitparams[2] is None: fitparams[2]=0
    if fitparams[2] is None: fitparams[2]=max_phase*180/np.pi
    if fitparams[3] is None: fitparams[3]=max(xdata) - min(xdata)
    if fitparams[4] is None: fitparams[4]=np.mean(ydata) #
    if fitparams[5] is None: fitparams[5]=xdata[0] # x0 (exp decay)
    if fitparams[6] is None: fitparams[6]=1 # y scale
    if fitparams[7] is None: fitparams[7]=1/10 # MHz
    if fitparams[8] is None: fitparams[8]=0 # phase degrees
    if fitparams[9] is None: fitparams[9]=0 # y0
    bounds = (
        [0.75*fitparams[0], 0.1/(max(xdata)-min(xdata)), -360, 0.3*(max(xdata)-min(xdata)), np.min(ydata), xdata[0]-(xdata[-1]-xdata[0]), 0.9, 0.01, -360, -0.1],
        [1.25*fitparams[0], 15/(max(xdata)-min(xdata)), 360, np.inf, np.max(ydata), xdata[-1]+(xdata[-1]-xdata[0]), 1.1, 10, 360, 0.1]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(twofreq_decaysin, xdata, ydata, p0=fitparams, bounds=bounds)
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov

# ====================================================== #
    
def hangerfunc(x, *p):
    f0, Qi, Qe, phi, scale, = p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    return scale * (1 - Q0/Qe * np.exp(1j*phi)/(1 + 2j*Q0*(x-f0)/f0))

def hangerS21func(x, *p):
    f0, Qi, Qe, phi, scale= p
    Q0 = 1 / (1/Qi + np.real(1/Qe))
    
    #return a0 + np.abs(hangerfunc(x, *p)) - scale*(1-Q0/Qe)
    return np.abs(hangerfunc(x, *p)) 

def hangerS21func_sloped(x, *p):
    f0, Qi, Qe, phi, scale, slope = p
    #return hangerS21func(x, f0, 1e4*Qi, 1e4*Qe, phi, scale)
    return hangerS21func(x, f0, 1e4*Qi, 1e4*Qe, phi, scale) + slope*(x-f0)

def hangerphasefunc(x, *p):
    return np.angle(hangerfunc(x, *p))

def fithanger(xdata, ydata, fitparams=None):
    # f0, Qi, Qe, phi, scale, a0, slope
    if fitparams is None: fitparams = [None]*6
    if fitparams[0] is None: 
        fitparams[0]=xdata[np.argmin(np.abs(ydata))] #f0
    if fitparams[1] is None: fitparams[1]=8 #Qi
    if fitparams[2] is None: fitparams[2]=3 #Qe
    if fitparams[3] is None: fitparams[3]=0 #phi
    if fitparams[4] is None: fitparams[4]=max(ydata) # scale
    if fitparams[5] is None: fitparams[5]=0 #(ydata[-1] - ydata[0]) / (xdata[-1] - xdata[0])
    #print(fitparams)

    # bounds = (
    #     [np.min(xdata), -1e9, -1e9, -2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))/10, -np.max(np.abs(ydata))],
    #     [np.max(xdata), 1e9, 1e9, 2*np.pi, (max(np.abs(ydata))-min(np.abs(ydata)))*10, np.max(np.abs(ydata))]
    #     )
    bounds = (
        [np.min(xdata),      0,      0, -np.inf, 0,       -np.inf],
        [np.max(xdata), np.inf, np.inf,  np.inf, np.inf,  np.inf],
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')

    pOpt = fitparams
    
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds)
        #print(pOpt)
        pOpt[1]=pOpt[1]
        pOpt[2]=pOpt[2]
        pOpt[0]=pOpt[0]
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        traceback.print_exc()
    #    pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov, fitparams

# ====================================================== #

def rb_func(depth, p, a, b):
    return a*p**depth + b

# Gives the average error rate over all gates in sequence
def rb_error(p, d): # d = dim of system = 2^(number of qubits)
    return 1 - (p + (1-p)/d)

# return covariance of rb error
def error_fit_err(cov_p, d):
    return cov_p*(1/d-1)**2

# Run both regular RB and interleaved RB to calculate this
def rb_gate_fidelity(p_rb, p_irb, d):
    return 1 - (d-1)*(1-p_irb/p_rb) / d

def fitrb(xdata, ydata, fitparams=None):
    if fitparams is None: fitparams = [None]*3
    if fitparams[0] is None: fitparams[0]=0.9
    if fitparams[1] is None: fitparams[1]=np.max(ydata) - np.min(ydata)
    if fitparams[2] is None: fitparams[2]=np.min(ydata)
    bounds = (
        [0, 0, 0],
        [1, 10*np.max(ydata)-np.min(ydata), np.max(ydata)]
        )
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.mean((bounds[0][i], bounds[1][i]))
            print(f'Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}')
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)
    try:
        pOpt, pCov = sp.optimize.curve_fit(rb_func, xdata, ydata, p0=fitparams, bounds=bounds)
        print(pOpt)
        print(pCov[0][0], pCov[1][1], pCov[2][2])
        # return pOpt, pCov
    except RuntimeError: 
        print('Warning: fit failed!')
        traceback.print_exc()
        pOpt = [np.nan]*len(pOpt)
        # return 0, 0
    return pOpt, pCov

# ====================================================== #
# Adiabatic pi pulse functions
# beta ~ slope of the frequency sweep (also adjusts width)
# mu ~ width of frequency sweep (also adjusts slope)
# period: delta frequency sweeps through zero at period/2
# amp_max

def adiabatic_amp(t, amp_max, beta, period):
    return amp_max / np.cosh(beta*(2*t/period - 1))

def adiabatic_phase(t, mu, beta, period):
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))

def adiabatic_iqamp(t, amp_max, mu, beta, period):
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j*np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j*np.cos(phase))
    return np.real(iamp), np.real(qamp)