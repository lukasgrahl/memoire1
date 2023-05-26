import os
import numpy as np
from scipy.stats import gamma, beta, norm, lognorm, uniform, invgamma

from settings import DATA_DIR

seed = sum(map(ord, "PyMC LABS - BVAR"))

# data to be pulled from FRED
fred_dict = {
    'y': ['GDP', "Gross Domestic Product"],
    'y_p': ['GDPPOT', 'Real Potential Gross Domestic Product'],
    'pi_s': ['CORESTICKM159SFRBATL', 'Sticky Price Consumer Price Index less Food and Energy'],
    'pi': ['MEDCPIM158SFRBCLE', 'Median Consumer Price Index (core inflation)'],
    'r': ['FEDFUNDS', 'Federal Funds Effective Rate'],
    'I': ['GPDI', 'Gross Private Domestic Investment'],
    'c_s': ['PCEPILFE', 'Personal Consumption Expenditures Excluding Food and Energy'],
    'c': ['PCE', 'Personal Consumption Expenditures'],
    'Ix': ['IMPGS', 'Imports of Goods and Services'],
    'Zx': ['EXPGS', 'Exports of Goods and Services'],
    'n': ['HOANBS', 'Nonfarm Business Sector: Hours Worked for All Workers'],
    'w': ['CES0500000003', 'Average Hourly Earnings of All Employees, Total Private'],
    'defl': ['A191RI1Q225SBEA', 'Gross Domestic Product: Implicit Price Deflator'],
    'recs': ['JHGDPBRINDX', 'GDP-Based Recession Indicator Index'],
    'Pop': ['CNP16OV', 'Population Level']
}

# time frame for FRED data
fred_start = "01/01/1970"
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
    'axes.spines.right': False,
    'font.size': 7,
}

# dictionary of recessions in the US economy, pulled from FRED through main.py
# made available for easy import here
recession_dict = np.load(os.path.join(DATA_DIR, 'recessions_periods.npy'))



## DSGE PARAMS
mod4_params = {
    'alpha': 0.35,
    'beta': 0.99,
    'delta': 0.02,
    'rho_A': 0.95,
    'sigma_C': 1.5,
    'sigma_L': 2.0,
    'sigma_epsilon_A': 0.05
}

mod4_priors = {
    'epsilon_a': beta(1.1, 10), # vasconez
    'alpha': uniform(0, .999), # del negro
    'delta': uniform(0, .999), # del negro
    'beta': norm(0.99, 0.01), # data
    'rho_A': beta(.5, .2), # vasconez
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
}

# gEconpy mod 5
mod5_params = {
    'alpha': 0.33,
    'beta': 0.99,
    'epsilon': 1.0,
    'sigma_L': 1.5,
    'phi_pi': 1.1,
    'phi_y': 0.4,
    'rho_v': 0.9,
    'rn': 0.01,
    'sigma_C': 1.5,
    'theta': 0.75
}

mod5_priors = {
    'alpha': beta(2, 5),
    'theta': beta(10, 3.4),
    'epsilon': beta(1.2, 1.2),
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
    'phi_pi': norm(1.2, .1), # vasconez
    'phi_y': norm(.5, .1), # vasconez
    'epsilon_v': beta(1.1, 10),
    'rho_v': beta(.5, .2), # vasconez
}

# model 6
mod6_params = {
    'epsilon': 1.5,
    'chi': 0.017,
    'alpha_m': 0.012,
    'alpha_n': 0.4,
    'beta': 0.999,
    'gamma': 0.1,
    'phi_pi': 1.1,
    'rho_s': 0.97,
    'theta': 0.75}

mod6_priors = {
    'alpha_m': beta(2, 5),
    'alpha_n': beta(2, 5),
    'epsilon': beta(1.2, 1.2),
    'theta': beta(.5, .1), # vasconez
    'phi_pi': norm(1.2, .1), # vasconez
    'sigma_C': norm(2, 2),
    'sigma_L': norm(2, 2),
    'chi': gamma(a=0.1, loc=0, scale=0.5),
    'epsilon_s': beta(1.1, 10),
    'rho_s': beta(.5, .2), # vasconez
}