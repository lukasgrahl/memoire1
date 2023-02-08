from settings import DATA_DIR

import os
import numpy as np

# data to be pulled from FRED
fred_dict = {
    'Y': 'GDP', # Gross Domestic Product
    'pi_s': 'CORESTICKM159SFRBATL', # Sticky Price Consumer Price Index less Food and Energy
    'pi_c': 'MEDCPIM158SFRBCLE', # Median Consumer Price Index (core inflation)
    'r': 'FEDFUNDS', # Federal Funds Effective Rate
    'I': 'GPDI', # Gross Private Domestic Investment
    'C': 'PCEPILFE', # Personal Consumption Expenditures Excluding Food and Energy
    'Ix': 'IMPGS', # Imports of Goods and Services
    'Zx': 'EXPGS', # Exports of Goods and Services
    'L': 'HOANBS', # Nonfarm Business Sector: Hours Worked for All Workers
    'w': 'CES0500000003', # Average Hourly Earnings of All Employees, Total Private
    'defl': 'A191RI1Q225SBEA', # Gross Domestic Product: Implicit Price Deflator
    'recs': 'JHGDPBRINDX', # GDP-Based Recession Indicator Index
}

fred_start = "01/01/1990"
fred_end = "01/01/2023"

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

recession_dict = np.load(os.path.join(DATA_DIR, 'recessions_periods.npy'))

