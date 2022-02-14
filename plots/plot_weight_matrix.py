
from plots.config_rcparams import *


def plot_weight_matrix(W):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    cax = ax.matshow(W)
    cbar = fig.colorbar(cax)
    cbar.set_label("$W_{ij}$", rotation=0, labelpad=10)
    plt.show()
