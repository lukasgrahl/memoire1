from settings import DATA_DIR

import os
import numpy as np

# data to be pulled from FRED
fred_dict = {
    'Y':   ['GDP', "Gross Domestic Product"],
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
    'figure.figsize':(10,3),
    'figure.dpi':144,
    'figure.facecolor':'white',
    'axes.grid':True,
    'grid.linestyle':'--',
    'grid.linewidth':0.5,
    'axes.spines.top':False,
    'axes.spines.bottom':False,
    'axes.spines.left':False,
    'axes.spines.right':False
}

# dictionary of recessions in the US economy, pulled from FRED through main.py
# made available for easy import here
recession_dict = np.load(os.path.join(DATA_DIR, 'recessions_periods.npy'))

## DSGE PARAMS

# gEconpy mod 5
mod5_params = {
    "sigma_C": 2,
    "sigma_L": 1.5,
    "alpha": .35,
    "beta": .985,
    "delta": .025,
    "rho_A": .95,
    "Theta": .75,
    "psi": 8
}
mod5_shocks = {
    "epsilon": .22
}

mod6_params = {
    "sigma_C": 1,
    "sigma_L": 1,
    "alpha": .35,
    "beta": .985,
    "delta": .025,
    "rho_A": .95,
    "Theta": .75,
    "Theta_w": .75,
    "psi": 8,
    "psi_w": 21
}


mod7_params = {
    'alpha': 0.35,
     'beta': 0.99,
     'delta': 0.025,
     'eta_p': 0.75,
     'eta_w': 0.75,
     'gamma_I': 10.0,
     'gamma_R': 0.9,
     'gamma_Y': 0.05,
     'gamma_pi': 1.5,
     'phi_H': 0.5,
     'psi_p': 0.6,
     'psi_w': 0.782,
     'rho_pi_dot': 0.924,
     'rho_preference': 0.95,
     'rho_technology': 0.95,
     'sigma_C': 2.0,
     'sigma_L': 1.5
}