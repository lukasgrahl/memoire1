import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
POST_EST_DIR = os.path.join(DATA_DIR, 'posterior_est_out')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model_files')
LATEX_OUT = os.path.join(PROJECT_ROOT, 'latex', 'graphs')

