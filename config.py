import yaml
from slab import AttrDict
from functools import reduce
import numpy as np

def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def load(file_name):
    with open(file_name,'r') as file:
      auto_cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 
    return auto_cfg

def save(cfg, file_name):
    # dump it: 
    cfg= yaml.safe_dump(cfg.to_dict(), default_flow_style=  None)

    # write it: 
    with open(file_name, 'w') as modified_file:
        modified_file.write(cfg)

    # now, open the modified file again 
    with open(file_name,'r') as file:
      cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 
    
    return cfg


def recursive_get(d, keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

def in_rng(val, rng_vals):
    if val < rng_vals[0]:
        print('Val is out of range, setting to min')
        return rng_vals[0]
    elif val > rng_vals[1]:
        print('Val is out of range, setting to max')
        return rng_vals[1]
    else:
        return val
    

def update_qubit(file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None):
    cfg=load(file_name)
    if not np.isnan(value):     
        if not isinstance(value, int):
            value=round(value, sig)
        if rng_vals is not None:
            value = in_rng(value, rng_vals)
        if isinstance(field, tuple): # for setting nested fields
            v=recursive_get(cfg['device']['qubit'], field)
            old_value=v[qubit_i]
            v[qubit_i]=value
            nested_set(cfg['device']['qubit'], field, v)
            if verbose: 
                print(f'*Set cfg qubit {qubit_i} {field} to {value} from {old_value}*')
        else:
            old_value = cfg['device']['qubit'][field][qubit_i]
            cfg['device']['qubit'][field][qubit_i] = value
            if verbose: 
                print(f'*Set cfg qubit {qubit_i} {field} to {value} from {old_value}*')
        save(cfg, file_name)

    return cfg 

def update_readout(file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None):
    cfg=load(file_name)
    if not np.isnan(value):
        if rng_vals is not None:
            value = in_rng(value, rng_vals)        
        value=round(value, sig)
        old_value = cfg['device']['readout'][field][qubit_i]
        cfg['device']['readout'][field][qubit_i] = value
        save(cfg, file_name)
        if verbose:
            print(f'*Set cfg resonator {qubit_i} {field} to {value} from {old_value}*')
    return cfg 

def init_config(file_name, num_qubits, type='full', t1=50, aliases='Qick001'):

    device={'qubit':{'pulses':{'pi_ge':{},'pi_ef':{}}}, 'readout':{}}
    #device = {'qubit': {'T1': [], 'f_ge': [], 'f_EgGf': [], 'f_ef': [], 'kappa': [], 'pulses': {'hpi_ge': {'gain': [], 'sigma': []}, 'pi_ge': {'gain': [], 'sigma': []}, 'pi_ef': {'gain': [], 'sigma': []}}}, 'readout': {'Max_amp': [], 'frequency': [], 'gain': [], 'phase': [], 'readout_length': [], 'threshold': [], 'kappa': []}}
    device['qubit']['temp'] = 0* num_qubits
    device['qubit']['T1']= [t1]*num_qubits
    device['qubit']['T2r']= [t1]*num_qubits
    device['qubit']['T2e']= [2*t1]*num_qubits
    device['qubit']['f_ge']= [4000]*num_qubits
    #device['qubit']['f_EgGf']= [2000]*num_qubits
    device['qubit']['f_ef']= [3800]*num_qubits
    device['qubit']['kappa']= [0]*num_qubits
    #device['qubit']['pulses']['hpi_ge']['gain']= [500]*num_qubits
    #device['qubit']['pulses']['hpi_ge']['sigma']= [0.20]*num_qubits
    #evice['qubit']['pulses']['hpi_ge']['type']= ['gauss']*num_qubits
    device['qubit']['pulses']['pi_ge']['gain']= [5000]*num_qubits
    device['qubit']['pulses']['pi_ge']['sigma']= [0.1]*num_qubits
    device['qubit']['pulses']['pi_ge']['type']= ['gauss']*num_qubits
    #device['qubit']['pulses']['pi_EgGf']['gain']= [10000]*num_qubits
    #device['qubit']['pulses']['pi_EgGf']['sigma']= [0.1]*num_qubits
    device['qubit']['pulses']['pi_ef']['type']= ['gauss']*num_qubits
    device['qubit']['pulses']['pi_ef']['gain']= [5000]*num_qubits
    device['qubit']['pulses']['pi_ef']['sigma']= [0.1]*num_qubits
    device['readout']['Max_amp']= [1]*num_qubits
    device['readout']['frequency']= [7000]*num_qubits
    device['readout']['gain']= [500]*num_qubits
    device['readout']['spec_gain']= [10]*num_qubits
    device['readout']['phase']= [0]*num_qubits
    device['readout']['readout_length']= [5]*num_qubits
    device['readout']['threshold']= [10]*num_qubits
    device['readout']['kappa']= [.5]*num_qubits
    device['readout']['trig_offset']= [150]*num_qubits
    device['readout']['relax_delay']= [1000]*num_qubits
    device['readout']['chi']= [0]*num_qubits
    device['readout']['fidelity']= [0]*num_qubits
    device['readout']['qe']= [0]*num_qubits
    device['readout']['qi']= [0]*num_qubits
    device['readout']['reps']= [1]*num_qubits
    device['readout']['rounds']= [1]*num_qubits


    soc = {'adcs':{'readout':{'ch':[0]*num_qubits}}, 'dacs':{'qubit':{'ch': [1]*num_qubits, 'nyquist':[1]*num_qubits, 'type':['full']*num_qubits}, 'readout':{'ch': [0]*num_qubits, 'nyquist':[2]*num_qubits, 'type':[type]*num_qubits}}}

    auto_cfg = {'device': device, 'hw':{'soc':soc}, 'aliases':{'soc': aliases}}

    cfg= yaml.safe_dump(auto_cfg, default_flow_style=  None)

    # write it: 
    with open(file_name, 'w') as modified_file:
        modified_file.write(cfg)

    return cfg