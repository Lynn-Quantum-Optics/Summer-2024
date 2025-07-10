import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties import core as ucore
from uncertainties import ufloat

# this file determines the angle chi of a given state sin(chi/2)|H>|alpha>+(e^i*gamma)cos(chi/2)|V>|alpha_perp>
# from the density matrix

def reformat_float_to_ufloat(df_:pd.DataFrame) -> pd.DataFrame:
        ''' Reformat a dataframe with floats to contain ufloats where applicable.

        Each column "X" that has a corresponding column "X_SEM" will be collapsed to the single column "X" containing ufloats.

        Parameters
        ----------
        df_ : pd.DataFrame
            The dataframe to reformat.
            
        Returns
        -------
        pd.DataFrame
            The reformatted dataframe.
        '''
        # create a copy of the dataframe to work with
        df = df_.copy()
        # loop through columns to see which should be reformatted
        to_reformat = [c for c in df.columns if c+'_SEM' in df.columns]
        # loop through columns to reformat
        for c in to_reformat:
            # recast to object type
            df[c] = df[c].astype(object)
            # create the ufloats
            for i in range(len(df)):
                df.at[i, c] = ufloat(df[c][i], df[c+'_SEM'][i])
        # drop the sem columns
        df.drop(columns=[c+'_SEM' for c in to_reformat], inplace=True)
        # return the new dataframe
        return df

def load_data(file_path:str) -> pd.DataFrame:
    ''' Load data saved by this class directly into a pandas dataframe.

    Parameters
    ----------
    file_path : str
        The path to the csv data file to load.
    
    Returns
    -------
    pd.DataFrame
        The data loaded from the file.
    '''
    # start by just loading the data
    df = pd.read_csv(file_path)
    # return a reformatted version with ufloats
    return reformat_float_to_ufloat(df)


if __name__ == '__main__':
    # first deg measurement, last deg measurement, # of steps, # of measurements per step, time per measurement
    DATE = "7022024" #please update
    # STATE = "ha_negpi_3_vd"
    chis_theo = np.linspace(0.001, np.pi/2, 6)
    chis_actual = []
    
    fileName = f"chi_data_from_tomo_summer_2025_analysis_hl_vr"
    directory = f'stu_hrvl'

    for chi in chis_theo:
        chi_theo = np.rad2deg(chi)
        datas = load_data(f'{directory}/tomo_data_{chi}_{DATE}.csv')

        datas = datas.set_index("note")

        # for hdva state: 
        chi_act = 2*unp.arctan(unp.sqrt(datas.at['VR', 'C4']/datas.at['HL', 'C4']))
        chis_actual.append(chi_act)

    df = pd.DataFrame({'Chi Theory': chis_theo, 'Chi Actual': chis_actual})

    pd.DataFrame(df).to_csv(f'{directory}/{fileName}.csv')

