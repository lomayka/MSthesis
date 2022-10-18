import math
import os
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def wind_data(path_name: str, random_state: int = 0, test_size:int=2*12*2):
    """
    Build the wind power data for the GEFcom IJF_paper case study.
    """

    df_wind = pd.read_csv(path_name, parse_dates=True, index_col=0)
    ZONES = ['ZONE_' + str(i) for i in range(1, 10 + 1)]

    # INPUTS DESCRIPTION
    # The predictors included wind forecasts at two heights, 10 and 100 m above ground level, obtained from the European Centre for Medium-range Weather Forecasts (ECMWF).
    # These forecasts were for the zonal and meridional wind components (denoted u and v), i.e., projections of the wind vector on the west-east and south-north axes, respectively.

    # U10 zonal wind component at 10 m
    # V10 meridional wind component at 10 m
    # U100 zonal wind component at 100 m
    # V100 meridional wind component at 100 m

    # ------------------------------------------------------------------------------------------------------------------
    # Build derived features
    # cf winner GEFcom2014 wind track “Probabilistic gradient boosting machines for GEFCom2014 wind forecasting”
    # ------------------------------------------------------------------------------------------------------------------

    # the wind speed (ws), wind energy (we), and wind direction (wd) were as follows,
    # where u and v are the wind components provided and d is the density, for which we used a constant 1.0
    # ws = sqrt[u**2  + v**2]
    # we = 0.5 × d × ws**3
    # wd = 180/π × arctan(u, v)

    df_wind['ws10'] = np.sqrt(df_wind['U10'].values ** 2 + df_wind['V10'].values ** 2)
    df_wind['ws100'] = np.sqrt(df_wind['U100'].values ** 2 + df_wind['V100'].values ** 2)
    df_wind['we10'] = 0.5 * 1 * df_wind['ws10'].values ** 3
    df_wind['we100'] = 0.5 * 1 * df_wind['ws100'].values ** 3
    df_wind['wd10'] = np.arctan2(df_wind['U10'].values, df_wind['V10'].values) * 180 / np.pi
    df_wind['wd100'] = np.arctan2(df_wind['U100'].values, df_wind['V100'].values) * 180 / np.pi

    features = ['U10', 'V10', 'U100', 'V100', 'ws10', 'ws100', 'we10', 'we100', 'wd10', 'wd100']

    data_zone = []
    for zone in ZONES:
        df_var = df_wind[df_wind[zone] == 1].copy()
        nb_days = int(len(df_var) / 24)
        zones = [df_var[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in ZONES]
        x = np.concatenate([df_var[col].values.reshape(nb_days, 24) for col in features] + zones, axis=1)
        y = df_var['TARGETVAR'].values.reshape(nb_days, 24)
        df_y = pd.DataFrame(data=y, index=df_var['TARGETVAR'].asfreq('D').index)
        df_x = pd.DataFrame(data=x, index=df_var['TARGETVAR'].asfreq('D').index)

        # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
        df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,random_state=random_state, shuffle=True)
        df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,random_state=random_state, shuffle=True)

        data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

        nb_days_LS = len(df_y_LS)
        nb_days_VS = len(df_y_VS)
        nb_days_TEST = len(df_y_TEST)
        print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return [pd.concat([data_zone[i][j] for i in range(0, 9 + 1)], axis=0, join='inner') for j in range(0, 5 + 1)]