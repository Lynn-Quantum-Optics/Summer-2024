# file to extract info from the random jones data
from os.path import join
import pandas as pd
import numpy as np

## load data generated from jones_datagen.py ##
DATA_PATH = 'jones_data'
df = pd.read_csv(join(DATA_PATH, 'jones_102000_0.csv'))

