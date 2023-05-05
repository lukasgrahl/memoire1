
from settings import DATA_DIR

import os
import numpy as np
from scipy.stats import gamma, beta, norm, lognorm

seed = sum(map(ord, "PyMC LABS - BVAR"))

# data to be pulled from FRED
fred_dict = {
    'Y': ['GDP', "Gross Domestic Product"],
    'Y_p': ['GDPPOT', 'Real Potential Gross Domestic Product'],
    'pi_s': ['CORESTICKM159SFRBATL', 'Sticky Price Consumer Price Index less Food and Energy'],
    'pi_c': ['MEDCPIM158SFRBCLE', 'Median Consumer Price Index (core inflation)'],
    'r': ['FEDFUNDS', 'Federal Funds Effective Rate'],
    'I': ['GPDI', 'Gross Private Domestic Investment'],
    'C': ['PCEPILFE', 'Personal Consumption Expenditures Excluding Food and Energy'],
    'Ix': ['IMPGS', 'Imports of Goods and Services'],
    'Zx': ['EXPGS', 'Exports of Goods and Services'],
    'L': ['HOANBS', 'Nonfarm Business Sector: Hours Worked for All Workers'],
    'w': ['CES0500000003', 'Average Hourly Earnings of All Employees, Total Private'],
    'defl': ['A191RI1Q225SBEA', 'Gross Domestic Product: Implicit Price Deflator'],
    'recs': ['JHGDPBRINDX', 'GDP-Based Recession Indicator Index'],
    'Pop': ['CNP16OV', 'Population Level']
}

# time frame for FRED data
fred_start = "01/01/1975"
fred_end = "12/31/2022"

# matplotlib config
plt_config = {
    'figure.figsize': (10, 3),
    'figure.dpi': 144,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False
}

# dictionary of recessions in the US economy, pulled from FRED through main.py
# made available for easy import here
recession_dict = np.load(os.path.join(DATA_DIR, 'recessions_periods.npy'))



## DSGE PARAMS

mod4_params = {'alpha': 0.35,
               'beta': 0.99,
               'delta': 0.02,
               'rho_A': 0.95,
               'sigma_C': 1.5,
               'sigma_L': 2.0,
               'sigma_epsilon_A': 0.05
               }

mod4_priors = {
    'alpha': beta(2, 5),
    'epsilon_A': beta(1.1, 10),
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
}

# gEconpy mod 5
mod5_params = {
    'alpha': 0.35,
    'beta': 0.99,
    'eta_p': 0.75,
    'gamma_R': 0.9,
    'gamma_Y': 0.05,
    'gamma_pi': 1.5,
    'psi_p': 0.6,
    'rho_A': 0.95,
    'rho_pi_dot': 0.924,
    'sigma_C': 1.5,
    'sigma_L': 2.0
}

mod5_priors = {
    'alpha': beta(2, 5),
    'eta_p': beta(10, 3.4),
    'gamma_R': gamma(4, 0, .5),
    'gamma_Y': gamma(4, 0, .5),
    'gamma_pi': gamma(4, 0, .5),
    'epsilon_A': beta(1.1, 10),
    'epsilon_R': beta(1.1, 10),
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
    'epsilon_T': beta(1.1, 10),
    'epsilon_Y': beta(1.1, 10),
    'epsilon_pi': beta(1.1, 10)
}

# model 6
mod6_params = {
    'M': 3.0,
    'X': 0.03,
    'alpha_m': 0.4,
    'alpha_n': 0.4,
    'beta': 0.995,
    'epsilon': 1.0,
    'gamma': 0.1,
    'phi_pi': 1.1,
    'rho_s': 0.9,
    'theta': 0.75}

mod6_priors = {
    'alpha_m': beta(2, 5),
    'alpha_n': beta(2, 5),
    'M': lognorm(scale=7.38905609893065, s=0.7),
    'gamma_pi': gamma(4, 0, .5),
    'epsilon': beta(1.2, 1.2),
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
    'epsilon_s': beta(1.1, 10),
}
