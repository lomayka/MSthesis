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

def res_model(s_model: dict, n_s: int, i: int, prices: dict, s_obs: dict, curtail:bool=True,  soc_max:float=0):
    """
    Compute the planning and dispatching for a given set of senarios.
    :param s_model: PV, wind, and load scenarios from a model, s_model['pv'] is of shape (#TEST, 24, 100)
    :param n_s: number of scenarios to select for optimization <= 100.
    :param i: day index.
    :param prices:
    :param s_obs: PV, wind, and load observations for the corresponding day.
    :return: dispatching solution into a list in keuros
    """
    n_s = min(100, n_s)
    scenarios = dict()
    scenarios['PV'] = s_model['pv'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i
    scenarios['W'] = s_model['wind'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i
    scenarios['L'] = s_model['load'][i,:,:n_s].transpose() # shape = (n_s, 24) for the day i

    # planner = Planner_dad(scenarios=scenarios, prices=prices, curtail=curtail)
    planner = Planner(scenarios=scenarios, prices=prices, curtail=curtail, soc_max=soc_max)

    planner.solve()
    sol = planner.store_solution()

    dis = Planner(scenarios=s_obs, prices=prices, x=sol['x'], curtail=curtail, soc_max=soc_max)
    dis.solve()
    sol_dis = dis.store_solution()

    return [sol_dis['obj'] / 1000, sol_dis['dad_profit'] / 1000, sol_dis['short_penalty'] / 1000, sol_dis['long_penalty'] / 1000, sol_dis['x']]

def energy_score(s: np.array, y_true: np.array):
    """
    Compute the Energy score (ES).
    :param s: scenarios of shape (24*n_days, n_s)
    :param y_true: observations of shape = (n_days, 24)
    :return: the ES per day of the testing set.
    """
    n_periods = y_true.shape[1]
    n_d = len(y_true)  # number of days
    n_s = s.shape[1]  # number of scenarios per day
    es = []
    # loop on all days
    for d in range(n_d):
        # select a day for both the scenarios and observations
        s_d = s[n_periods * d:n_periods * (d + 1), :]
        y_d = y_true[d, :]

        # compute the part of the ES
        simple_sum = np.mean([np.linalg.norm(s_d[:, s] - y_d) for s in range(n_s)])

        # compute the second part of the ES
        double_somme = 0
        for i in range(n_s):
            for j in range(n_s):
                double_somme += np.linalg.norm(s_d[:, i] - s_d[:, j])
        double_sum = double_somme / (2 * n_s * n_s)

        # ES per day
        es_d = simple_sum - double_sum
        es.append(es_d)
    return es