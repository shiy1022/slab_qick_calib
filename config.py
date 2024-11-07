import yaml
from slab import AttrDict
from functools import reduce


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

def update_qubit(file_name, field, value, qubit_i, verbose=True):
    cfg=load(file_name)
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

def update_readout(file_name, field, value, qubit_i, verbose=True):
    cfg=load(file_name)
    old_value = cfg['device']['readout'][field][qubit_i]
    cfg['device']['readout'][field][qubit_i] = value
    save(cfg, file_name)

    print(f'*Set cfg resonator {qubit_i} {field} to {value} from {old_value}*')
    return cfg 

def init_config(file_name, num_qubits, aliases='Qick001'):

    device={}
    device = {'qubit': {'T1': [], 'f_ge': [], 'f_EgGf': [], 'f_ef': [], 'kappa': [], 'pulses': {'hpi_ge': {'gain': [], 'sigma': []}, 'pi_ge': {'gain': [], 'sigma': []}, 'pi_EgGf': {'gain': [], 'sigma': []}, 'pi_ef': {'gain': [], 'sigma': []}}}, 'readout': {'Max_amp': [], 'frequency': [], 'gain': [], 'phase': [], 'readout_length': [], 'threshold': [], 'kappa': []}}
    device['qubit']['T1']= [100.0]*num_qubits
    device['qubit']['T2r']= [100.0]*num_qubits
    device['qubit']['T2e']= [200.0]*num_qubits
    device['qubit']['f_ge']= [0.1]*num_qubits
    #device['qubit']['f_EgGf']= [2000]*num_qubits
    device['qubit']['f_ef']= [0.1]*num_qubits
    device['qubit']['kappa']= [0]*num_qubits
    #device['qubit']['pulses']['hpi_ge']['gain']= [500]*num_qubits
    #device['qubit']['pulses']['hpi_ge']['sigma']= [0.20]*num_qubits
    d#evice['qubit']['pulses']['hpi_ge']['type']= ['gauss']*num_qubits
    device['qubit']['pulses']['pi_ge']['gain']= [0.05]*num_qubits
    device['qubit']['pulses']['pi_ge']['sigma']= [0.1]*num_qubits
    device['qubit']['pulses']['pi_ge']['type']= ['gauss']*num_qubits
    #device['qubit']['pulses']['pi_EgGf']['gain']= [10000]*num_qubits
    #device['qubit']['pulses']['pi_EgGf']['sigma']= [0.1]*num_qubits
    device['qubit']['pulses']['pi_ef']['type']= ['gauss']*num_qubits
    device['qubit']['pulses']['pi_ef']['gain']= [0.3]*num_qubits
    device['qubit']['pulses']['pi_ef']['sigma']= [0.1]*num_qubits
    device['readout']['Max_amp']= [1]*num_qubits
    device['readout']['frequency']= [7000]*num_qubits
    device['readout']['gain']= [0.5]*num_qubits
    device['readout']['phase']= [0]*num_qubits
    device['readout']['readout_length']= [5]*num_qubits
    device['readout']['threshold']= [10]*num_qubits
    device['readout']['kappa']= [0]*num_qubits
    device['readout']['trig_offset']= [150]*num_qubits
    device['readout']['relax_delay']= [1000]*num_qubits
    device['readout']['chi']= [0]*num_qubits

    chans = {'ch': [0]*num_qubits, 'nyquist':[2]*num_qubits, 'type':['full']*num_qubits}
    soc = {'adcs':{'readout':{'ch':[0]*num_qubits}}, 'dacs':{'qubit':{'ch': [0]*num_qubits, 'nyquist':[1]*num_qubits, 'type':['full']*num_qubits}, 'readout':{'ch': [0]*num_qubits, 'nyquist':[2]*num_qubits, 'type':['full']*num_qubits}}}

    auto_cfg = {'device': device, 'hw':{'soc':soc}, 'aliases':aliases}

    # # dump it: 
    auto_cfg= yaml.safe_dump(auto_cfg, default_flow_style=None)
    #auto_cfg= yaml.safe_dump(auto_cfg, canonical=True)

    # # write it: 
    with open(file_name, 'w') as modified_file:
        modified_file.write(auto_cfg)

    # now, open the modified file again 
    with open(file_name,'r') as file:
        auto_cfg=AttrDict(yaml.safe_load(file)) # turn it into an attribute dictionary 