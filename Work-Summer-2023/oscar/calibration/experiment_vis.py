# file to fit data from the experiment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from scipy.optimize import curve_fit, minimize
from scipy.stats import sem

DATA_PATH = '../Experiment'

def plot_fit_all(fi_ls, do, loadpath, title_text, figname,savepath): # name is what you want to name the output image; do is list of binaries matching the fi_ls: 0 means find min, 1 finds max; fi is fit index; that is, which graph to fit
    df = pd.read_csv(loadpath)
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(7, 8))
    axes[0].set_title(title_text, fontsize=16)
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

def plot_fit_C4(loadpath, filename, title_text, figname):
    df = pd.read_csv(join(loadpath, filename))
    Theta = np.array(df['Theta'].values)
    HH = np.array(df['HH'].values)
    HH_unc = np.array(df['HH_unc'].values)
    VV = np.array(df['VV'].values)
    VV_unc = np.array(df['VV_unc'].values)

    stats = pd.DataFrame({'meas':[], 'A': [], 'Aerr':[], 'c':[],'cerr':[],'d':[],'derr':[], 'chi2red':[], 'e_val':[], 'e_sem':[], 'frac_1s':[]})


    fig, axes=plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    axes[0].errorbar(Theta, HH, yerr=HH_unc, fmt='o', label='HH counts')
    axes[0].errorbar(Theta, VV, yerr=VV_unc, fmt='o', label='VV counts')
    axes[0].legend()
    angles = np.linspace(min(df['Theta'].values), max(df['Theta'].values), 1000)
    def fit_func(x, A, c, d):
            return A*(np.sin(2*x - c))**2+d
    def fit_to_func(Theta, meas, unc, meas_name):
        popt, pcov = curve_fit(fit_func, Theta, meas, sigma = unc, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))

        y_pred = fit_func(Theta, *popt)
        norm_resid = (meas - y_pred) / unc
        chi2red = sum(norm_resid**2) / (len(df['Theta'].values) - len(popt))

        # find min/max
        # min_val = minimize(fit_func, x0=min(df['Theta'].values), args=(*popt,)).x[0]
        ordered_indices = np.argsort(fit_func(angles[angles < np.pi/2], *popt))
        ordered = angles[ordered_indices]
        ordered_cut=ordered[-20:] # get max
        e_val = np.mean(ordered_cut) # take average of first 10 data points in ordered array; this is min_val
        e_sem = sem(ordered_cut)

        # get fraction of norm resid within +- 1 sigma
        norm_resid_1s = norm_resid[abs(norm_resid) <= 1]
        frac_1s = len(norm_resid_1s) / len(norm_resid)

        info =  {'meas':meas_name, 'A': popt[0], 'Aerr':perr[0], 'c':popt[1],'cerr':perr[1],'d':popt[2],'derr':perr[2], 'chi2red':chi2red, 'e_val':e_val, 'min_sem':e_sem, 'frac_1s':frac_1s}
        return norm_resid, popt, info
    norm_resid_HH, poptHH, info_HH = fit_to_func(Theta, HH, HH_unc, 'HH')
    norm_resid_VV, poptVV, info_VV = fit_to_func(Theta, VV, VV_unc, 'VV')

    # find angle where they equal
    HH_fit = fit_func(angles, *poptHH)
    VV_fit = fit_func(angles, *poptVV)
    equal= angles[abs(HH_fit - VV_fit) < .01]
    print(equal)

    stats = stats.append(info_HH, ignore_index=True)
    stats = stats.append(info_VV, ignore_index=True)
    stats.to_csv(join(loadpath, figname+'.csv'))

    axes[0].plot(angles,fit_func(angles, *poptHH), label='HH fit')
    axes[0].plot(angles,fit_func(angles, *poptVV), label='VV fit')
    axes[0].set_ylabel('Counts', fontsize=14)
    axes[0].legend()

    axes[1].errorbar(Theta, norm_resid_HH, yerr=np.ones(len(norm_resid_HH)), fmt='o',label='HH')
    axes[1].errorbar(Theta, norm_resid_VV, yerr=np.ones(len(norm_resid_VV)), fmt='o',label='VV')
    axes[1].set_ylabel('Normalized Residuals', fontsize=14)
    axes[1].legend()

    plt.xlabel('$\\theta$', fontsize=14)
    axes[0].set_title(title_text, fontsize=16)
    plt.savefig(join(loadpath, figname+'.pdf'))
    plt.show()


# plot_fit(fi_ls=[1,2], do=[0,1], loadpath=join(DATA_PATH, '5-31', '05-31-2023_17-10-14__5__frthet120_ph0__tothet131_ph0__n20', '05-31-2023_17-10-14__SUMMARY.csv'), figname='5_31_trial1', savepath=join(DATA_PATH, '5-31'))
# plot_fit_all(fi_ls=np.arange(1, 6, 1), do=np.ones(5), title_text='6/01/23 HH Counts', loadpath=join(DATA_PATH, '6-01', '06-01-2023_13-06-55__1__frthet0_ph0__tothet3
plot_fit_C4(loadpath=join(DATA_PATH, '6-01'), filename='uv_hwp_0.csv', title_text='UV HWP Callibration, using C4', figname='UV_HWP_fit_0')