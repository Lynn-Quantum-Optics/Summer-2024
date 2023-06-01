# file to fit data from the experiment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from scipy.optimize import curve_fit, minimize
from scipy.stats import sem

DATA_PATH = '../Experiment'

def plot_fit(fi_ls, do, loadpath, figname,savepath ): # name is what you want to name the output image; do is list of binaries matching the fi_ls: 0 means find min, 1 finds max; fi is fit index; that is, which graph to fit
    df = pd.read_csv(loadpath)
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(7, 8))
    axes[0].set_title('5/31/23 Run 1', fontsize=16)
    axes[0].errorbar(df['Theta'].values, df['C0 (A)'].values, yerr=df['C0 (A) uncertainty'], fmt='o', label='C0', color='red')
    axes[0].legend()
    axes[1].errorbar(df['Theta'].values, df['C1 (B)'].values, yerr=df['C1 (B) uncertainty'], fmt='o',label='C1', color='green')
    axes[1].legend()
    axes[2].errorbar(df['Theta'].values, df["C3 (B')"].values, yerr=df["C3 (B') uncertainty"], fmt='o',label='C3', color='blue')
    axes[2].legend()
    axes[3].errorbar(df['Theta'].values, df["C4"].values, yerr=df["C4 uncertainty"], fmt='o',label='C4', color='gold')
    axes[3].legend()
    axes[4].errorbar(df['Theta'].values, df["C6"].values, yerr=df["C6 uncertainty"], fmt='o',label='C6', color='indigo')
    axes[4].legend()
    plt.xlabel('$\\theta$', fontsize=14)

    # go back and fit
    stats = pd.DataFrame({'fi':[], 'A': [], 'Aerr':[], 'c':[],'cerr':[],'d':[],'derr':[], 'chi2red':[], 'e_val':[], 'e_sem':[], 'frac_1s':[]})
    c_vals = ['C0 (A)', 'C1 (B)', "C3 (B')", "C4", "C6"]
    norm_resid_ls = []
    for j, fi in enumerate(fi_ls):
        if fi >= 0:
            def fit_func(x, A, c, d):
                return A*(np.sin(x - c))**2+d

            popt, pcov = curve_fit(fit_func, np.array(df['Theta'].values), np.array(df[c_vals[fi]].values), sigma = np.array(df[c_vals[fi]+' uncertainty'].values), absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            angles = np.linspace(min(df['Theta'].values), max(df['Theta'].values), 1000)
            axes[fi].plot(angles, fit_func(angles, *popt))

            y_actual = np.array(df[c_vals[fi]].values)
            y_pred = fit_func(np.array(df['Theta'].values), *popt)
            y_err = np.array(df[c_vals[fi]+' uncertainty'].values)

            norm_resid = (y_actual - y_pred) / y_err
            norm_resid_ls.append(norm_resid)
            chi2red = sum(norm_resid**2) / (len(df['Theta'].values) - len(popt))

            # find min/max
            # min_val = minimize(fit_func, x0=min(df['Theta'].values), args=(*popt,)).x[0]
            ordered_indices = np.argsort(fit_func(angles, *popt))
            ordered = angles[ordered_indices]
            if do[j]==0:
                ordered_cut=ordered[:20]
            else:
                ordered_cut=ordered[-20:]
            e_val = np.mean(ordered_cut) # take average of first 10 data points in ordered array; this is min_val
            e_sem = sem(ordered_cut)

            # get fraction of norm resid within +- 1 sigma
            norm_resid_1s = norm_resid[abs(norm_resid) <= 1]
            frac_1s = len(norm_resid_1s) / len(norm_resid)

            stats =stats.append({'fi':fi, 'A': popt[0], 'Aerr':perr[0], 'c':popt[1],'cerr':perr[1],'d':popt[2],'derr':perr[2], 'chi2red':chi2red, 'e_val':e_val, 'min_sem':e_sem, 'frac_1s':frac_1s}, ignore_index=True)

    print(stats)
    plt.savefig(join(savepath, figname+'.pdf'))
    plt.show()
    stats.to_csv(join(savepath, figname+'.csv'))
    
    if len(fi_ls)>1:
        fig, axes = plt.subplots(len(fi_ls), 1, sharex=True,figsize=(7,5))
        for i, norm_resid in enumerate(norm_resid_ls):
            axes[i].set_title('Normalized Residuals for %i'%fi_ls[i], fontsize=16)
            axes[i].errorbar(np.array(df['Theta'].values), norm_resid, yerr=np.ones(len(norm_resid)), fmt='o')
        plt.xlabel('$\Theta$', fontsize=14)
        plt.savefig(join(savepath, figname+'_norm_resid.pdf'))
        plt.show()
    else:
        plt.figure(figsize=(10,7))
        plt.errorbar(np.array(df['Theta'].values), norm_resid, yerr=np.ones(len(norm_resid)), fmt='o')
        plt.xlabel('$\Theta$', fontsize=14)
        plt.title('Normalized Residuals for %i'%fi, fontsize=16)
        plt.savefig(join(savepath, figname+'_norm_resid.pdf'))
        plt.show()

plot_fit(fi_ls=[1,2], do=[0,1], loadpath=join(DATA_PATH, '5-31', '05-31-2023_17-10-14__5__frthet120_ph0__tothet131_ph0__n20', '05-31-2023_17-10-14__SUMMARY.csv'), figname='5_31_trial1', savepath=join(DATA_PATH, '5-31'))