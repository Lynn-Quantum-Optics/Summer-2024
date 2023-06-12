import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def acquire_data(all_data_file:str):
    # load data and initialize output
    all_data = pd.read_csv(all_data_file)
    new_data = {'time (min)':[], 'HH':[], 'VV':[], 'HH_err':[], 'VV_err':[]}

    # loop to re-format data
    for i in range(len(all_data)):
        new_data['time (min)'].append(i)
        if i % 2 == 0:
            # HH data
            new_data['HH'].append(all_data.iloc[i]['C4 rate (#/s)'])
            new_data['VV'].append(None)
            new_data['HH_err'].append(all_data.iloc[i]['C4 rate SEM (#/s)'])
            new_data['VV_err'].append(None)
        else:
            # VV data
            new_data['HH'].append(None)
            new_data['VV'].append(all_data.iloc[i]['C4 rate (#/s)'])
            new_data['HH_err'].append(None)
            new_data['VV_err'].append(all_data.iloc[i]['C4 rate SEM (#/s)'])

    return pd.DataFrame(new_data)

def aggregate_data(df:pd.DataFrame, num_samp:int, n_trial:int) -> pd.DataFrame:
    ''' Aggregates drift experiment data every num_samp data points.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing drift experiment data
    num_samp : int
        Number of samples of both HH and VV to aggregate over. Since the experiment takes an HH or VV measurement every minute, the interval of the resulting dataframe will be 2*num_samp minutes.
    n_trial : int
        The number of trials that the input dataframe sampled for each count rate. This affects the calculation of a cummulative SEM for data points.
    
    Returns
    -------
    pd.DataFrame
        The output data from the experiment.
    '''
    # calculate new number of trials for output SEM
    n_trial_new = num_samp * n_trial
    sem_factor = np.sqrt((n_trial * (n_trial-1))/(n_trial_new * (n_trial_new - 1)))

    # initialize output dataframe
    out_df = {'time (min)': [], 'HH': [], 'HH_err': [], 'VV': [], 'VV_err': []}

    # loop to fill in the dataframe
    for i in np.arange(0, len(df) - 2*num_samp + 1, 2 * num_samp):
        out_df['time (min)'].append(i)
        out_df['HH'].append(df['HH'][i:i+2*num_samp].mean())
        out_df['VV'].append(df['VV'][i:i+2*num_samp].mean())
        out_df['HH_err'].append(df['HH_err'][i:i+2*num_samp].sum() * sem_factor)
        out_df['VV_err'].append(df['VV_err'][i:i+2*num_samp].sum() * sem_factor)
    
    return pd.DataFrame(out_df)

def calculate_percentages(df:pd.DataFrame):
    ''' Uses HH and VV detections at every time stamp to compile percentage statistics.
    These values are filled in in the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing drift experiment data
    '''
    df['pct HH'] = 100 * df['HH']/(df['HH'] + df['VV'])
    df['pct HH err'] = 100 * np.sqrt((df['HH']*df['VV_err'])**2 + (df['VV']*df['HH_err'])**2) / (df['HH'] + df['VV'])**2
    return df

if __name__ == '__main__':
    # parameters
    
    PLOT_TITLE = 'Ratio of HH to HH and VV detections in the $\Phi^+$ state\nData taken in each basis for 30 seconds, every other minute for 4 hours'
    ALL_DATA_FILE = 'drift_experiment_all_data.csv'
    OUTPUT_FILE = 'DE_output.csv'
    AGGREGATE_EVERY = 10 # minutes
    NUM_SAMP_ORIGINAL = 6 # number of samples in original data collection

    df = acquire_data(ALL_DATA_FILE)
    df = aggregate_data(df, AGGREGATE_EVERY // 2, NUM_SAMP_ORIGINAL)
    df = calculate_percentages(df)

    # save the output data frame before plotting
    df.to_csv(OUTPUT_FILE, index=False)

    # plot the data
    plt.errorbar(x=df['time (min)'], y=df['pct HH'], yerr=df['pct HH err'], ms=5, fmt='ko-')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Percentage of HH detections\n(of HH and VV)')
    plt.title(PLOT_TITLE)
    plt.show()
