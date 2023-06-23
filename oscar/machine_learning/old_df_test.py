# file to read in all dfs from BL last semester

import pandas as pd
import os

df_path_main = 'S22_data'

df_path_ls = []
df = pd.DataFrame()
for path in os.listdir(df_path_main):
    if path.endswith('.csv'):
        df = pd.concat([df, pd.read_csv(os.path.join(df_path_main, path))])
