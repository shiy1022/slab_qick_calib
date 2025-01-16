from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
def plot_many(d,title='', save_path=None, chan='avgi',yax=None, norm=False, horz_line=None):
    fig, ax = plt.subplots(4,5, figsize=(20,16))
    ax = ax.flatten()
    fig.suptitle(title)
    for i in range(20):
        if norm==True: 
            for j in range(len(d[i].data[chan])):
                d[i].data[chan][j,:] = d[i].data[chan][j,:] / np.median(d[i].data[chan][j,:])
            
        ax[i].pcolormesh(d[i].data['xpts'],d[i].data['ypts'],d[i].data[chan])
        ax[i].set_title(f"Q{i}")
        if horz_line is not None:
            ax[i].axhline(horz_line[i], color='red')
        if yax=='log':
            ax[i].set_yscale('log')
    
    fig.tight_layout()
    if save_path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = save_path +'images\\summary\\'+ title + "_" + current_time + ".png"
        plt.savefig(fname)
