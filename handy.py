from datetime import datetime
import matplotlib.pyplot as plt

def plot_many(d,title='', save_path =None):
    fig, ax = plt.subplots(5,4, figsize=(20,20))
    ax = ax.flatten()
    fig.suptitle(title)
    for i in range(20):
        ax[i].pcolormesh(d[i].data['xpts'],d[i].data['ypts'],d[i].data['avgi'])
        ax[i].set_title(f"Q{i}")
    fig.tight_layout()
    if save_path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = save_path +'images\\summary\\'+ title + "_" + current_time + ".png"
        plt.savefig(fname)
