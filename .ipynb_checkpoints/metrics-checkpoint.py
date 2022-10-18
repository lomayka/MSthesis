import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def plf_per_quantile(quantiles:np.array, y_true:np.array):
    """
    Compute PLF per quantile.
    :param quantiles: (nb_periods, nb_quantiles)
    :param y_true:  (nb_periods,)
    :return: PLF per quantile into an array (nb_quantiles, )
    """
    # quantile q from 0 to N_q -> add 1 to be from 1 to N_q into the PLF score
    N_q = quantiles.shape[0]
    plf = []
    for q in range(0 ,N_q):
        # for a given quantile compute the PLF over the entire dataset
        diff = y_true - quantiles[q,:]
        
        plf_q = sum(diff[diff >= 0] * ((q+1) / (N_q+1))) / len(diff) + sum(-diff[diff < 0] * (1 - (q+1) / (N_q+1))) / len(diff) # q from 0 to N_q-1 -> add 1 to be from 1 to N_q
        plf.append(plf_q)
    return 100 * np.asarray(plf)

def plot_plf_per_quantile(plf_VS: np.array, plf_TEST: np.array, dir_path: str, name: str, ymax:float=None):
    """
    Plot the quantile score (PLF = Pinball Loss Function) per quantile on the VS & TEST sets.
    """
    FONTSIZE = 10
    plt.figure()
    plt.plot([q for q in range(1, len(plf_VS) + 1)], plf_TEST, 'b')
    plt.hlines(y=plf_TEST.mean(), colors='b', xmin=1, xmax=len(plf_VS),  label='TEST av ' + str(round(plf_TEST.mean(), 4)))
    plt.plot([q for q in range(1, len(plf_VS) + 1)], plf_VS, 'g')
    plt.hlines(y=plf_VS.mean(), colors='g', xmin=1, xmax=len(plf_VS), label='VS av ' + str(round(plf_VS.mean(), 4)))
    if ymax:
        plt.ylim(0, ymax)
        plt.vlines(x=(len(plf_VS) + 1) / 2, colors='k', ymin=0, ymax=ymax)
    else:
        plt.ylim(0, max(plf_TEST.max(), plf_VS.max()))
        plt.vlines(x=(len(plf_VS) + 1) / 2, colors='k', ymin=0, ymax=max(plf_TEST.max(), plf_VS.max()))
    plt.xlim(0, len(plf_VS) + 1)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlabel('q', fontsize=FONTSIZE)
    plt.ylabel('%', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dir_path + name + '.pdf')
    plt.show()