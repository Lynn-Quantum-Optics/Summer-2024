import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core import analysis

def acquire_data(all_data_file:str):
    # load data and initialize output
    all_data = pd.read_csv(all_data_file)
    new_data = {'time (min)':[], 'HH':[], 'VV':[], 'HH_err':[], 'VV_err':[]}

    # get start and stop times
    start_time = all_data.iloc[0]['start time (s)']
    end_time = all_data.iloc[-1]['stop time (s)']

    # get the samples per data point
    num_samp_orig = all_data.iloc[0]['num samples (#)']

    # collection period
    collection_period = all_data.iloc[0]['period per sample (s)'] * num_samp_orig

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

    return pd.DataFrame(new_data), start_time, end_time, num_samp_orig, collection_period

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
    # obtain the data
    ALL_DATA_FILE = 'drift_experiment/DE2_all_data.csv'
    df, start_time, end_time, num_samp_orig, collection_period = acquire_data(ALL_DATA_FILE)

    # other parameters
    AGGREGATE_EVERY = 2
    EXPERIMENT_LENGTH = int(df.iloc[-1]['time (min)'] / 60) # hours
    OUTPUT_PREFIX = 'DE2'
    PLOT_TITLE = f'HH and VV detections of the $\Phi^+$ state ({start_time} - {end_time})\nData taken in each basis for {collection_period} seconds, every other minute for {EXPERIMENT_LENGTH} hours (avg. every {AGGREGATE_EVERY} min)'

    # aggregate the data
    df = aggregate_data(df, AGGREGATE_EVERY // 2, num_samp_orig)
    df = calculate_percentages(df)

    # df = df.iloc[:len(df)//3]
    # df = df.iloc[len(df)//3:2*len(df)//3]
    df = df.iloc[2*len(df)//3:]

    # fit a linear model to the data
    params, errs = analysis.fit('line', df['time (min)'], df['pct HH'], df['pct HH err'])

    # put together the plots
    fig = plt.figure(figsize=(10, 10))

    # HH and VV data together
    ax = fig.add_subplot(411)
    ax.errorbar(x=df['time (min)'], y=df['HH']+df['VV'], yerr=df['HH_err'] + df['VV_err'], fmt='bo-', ms=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Count Rate (#/s)')
    ax.set_title('Cummulative HH + VV Count Rate Over Time')

    # VV - HH
    ax = fig.add_subplot(412)
    ax.errorbar(x=df['time (min)'], y=df['VV'] - df['HH'], yerr=df['HH_err'] + df['VV_err'], fmt='bo-', ms=2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Count Rate (#/s)')
    ax.set_title('Difference in Count Rates (VV - HH) Over Time')

    # HH and VV data separately
    ax = fig.add_subplot(413)
    ax.errorbar(x=df['time (min)'], y=df['HH'], yerr=df['HH_err'], fmt='go-', ms=2, label='HH')
    ax.errorbar(x=df['time (min)'], y=df['VV'], yerr=df['VV_err'], fmt='bo-', ms=2, label='VV')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Count Rate (#/s)')
    ax.set_title('HH and VV Count Rates Over Time')
    ax.legend()
    
    # percentage data
    ax = fig.add_subplot(414)
    ax.errorbar(x=df['time (min)'], y=df['pct HH'], yerr=df['pct HH err'], ms=2, fmt='bo-')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Percentage of HH detections (%)')
    ax.set_title('Percentage of HH Detections Over Time')
    analysis.plot_func('line', params, df['time (min)'], ax, label=f'Linear Fit (slope = {params[0]*60:.5f}±{errs[0]*60:.5f} %/hr)', color='r')
    ax.legend()

    # overall title
    fig.suptitle(PLOT_TITLE)
    fig.tight_layout()
    
    # save the plot
    plt.savefig(f'{OUTPUT_PREFIX}_plot.png', dpi=600)
