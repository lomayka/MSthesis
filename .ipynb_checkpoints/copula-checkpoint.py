# These function are taken from Ilias Dimoulkas from Greenlytics. Ask for his permission to use them here!
# The implementation is based on this paper:
# Pinson et. al, From Probabilistic Forecasts to Statistical Scenarios of Short-term Wind Power Production (2008)

import numpy as np
import pandas as pd
import scipy as scp
import scipy.stats
from matplotlib import pyplot as plt


def update_cov_matrix(
    df_y_pred_q, quantiles, actual_data, forget_factor, cov_matrix, debug=False
):
    """
    df_y_pred_q: (n_period, n_quantiles) where n_period represents each time step on the horizon
    quantiles: (n_quantiles,) inquired quantiles from the model
    actual_data: ground truth
    forget_factor: 0 is maximum forgetting and closer to 1 is the opposite
    cov_matrix: estimated covariance matrix from previous step, for first step could be identity
    debug: if true covariance matrix is printed out
    """

    n_periods = np.size(df_y_pred_q, 0)

    # Eq. 3 - Probability Integral Transform
    Y = np.zeros(n_periods)
    for i in range(n_periods):
        Fhat = scp.interpolate.interp1d(
            df_y_pred_q.iloc[i],
            quantiles,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        Y[i] = min(
            0.99999, max(0.00001, Fhat(actual_data.iloc[i]))
        )  # Apply lower (0) and upper (1) limits

    # Eq. 5 - probit (inverse of Gaussian CDF)
    X = scp.stats.norm.ppf(Y)
    print(X)
    # Eq. 9 - Update covariance matrix
    cov_matrix = forget_factor * cov_matrix + (1 - forget_factor) * np.outer(X, X)

    # Eq. 10 - Normalize covariance matrix
    std_vector = np.sqrt(np.diagonal(cov_matrix))
    cov_matrix = cov_matrix / np.outer(std_vector, std_vector)

    if debug:
        print("covariance matrix")
        print(cov_matrix)

    return cov_matrix


def sample_from_quantiles(
    df_y_pred_q,
    quantiles,
    n_scenarios,
    cov_matrix=[],
    rnd_seed=[],
    sort="No",
    debug=False,
):
    """
    Obtain scenarios based on quantiles and estimated multivariate Gaussian covariance
    df_y_pred_q: (n_period, n_quantiles) where n_period represents each time step on the horizon
    quantiles: (n_quantiles,) inquired quantiles from the model
    n_scenarios: number of intended scenarios/samples
    cov_matrix: estimated covariance matrix
    rnd_seed:
    sort:
    debug: if true covariance matrix is printed out
    """
    n_periods = np.size(df_y_pred_q, 0)

    if np.size(cov_matrix, 0) == 0:
        cov_matrix = np.identity(n_periods)
    if np.size(rnd_seed, 0) != 0:
        np.random.seed(rnd_seed)

    mean_vector = np.zeros(n_periods)
    std_vector = np.sqrt(np.diagonal(cov_matrix))

    # Eq. 10 - Normalize covariance matrix
    cov_matrix = cov_matrix / np.outer(std_vector, std_vector)
    if debug:
        print("covariance matrix")
        print(cov_matrix)

    # step (i) - Generate multivariate Gaussian random numbers with zero mean and covariance matrix
    X = np.random.multivariate_normal(mean_vector, cov_matrix, n_scenarios)

    # Sort
    if sort == "Yes":
        X = np.sort(X, axis=0)

    # step (ii) - Gaussian --> Uniform transformation
    Y = scp.stats.norm.cdf(X)

    # Plot hist of X (normal) and Y(uniform)
    if debug:
        bin_cnt = 200
        plt.hist(X[:, 0], bin_cnt)
        plt.show()
        plt.hist(Y[:, 0], bin_cnt)
        plt.show()

    # step (iii) - Calculate interpolation function
    Finv = scp.interpolate.interp1d(
        quantiles,
        df_y_pred_q,
        kind="linear",
        axis=1,
        bounds_error=False,
        fill_value="extrapolate",
    )

    columns = ["scenario{0}".format(int(i)) for i in range(n_scenarios)]
    scenarios = pd.DataFrame(index=df_y_pred_q.index, columns=columns)
    for i in range(n_periods):
        scenarios.iloc[i] = Finv(Y[:, i])[i, :]

    # Print CDF and scenarios
    if debug:
        for i in range(n_periods):
            plt.plot(
                df_y_pred_q.iloc[i], quantiles, "o", scenarios.iloc[i], Y[:, i], "x"
            )
            plt.show()

    return scenarios
