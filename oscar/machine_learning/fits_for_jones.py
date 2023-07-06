# file to fit data to determine experimental corrections #

from os.path import join
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def det_a():
    '''Function to determine te fraction max VV / max HH counts using Richard's QP sweep.'''

    # read in data #
    df_vv = pd.read_csv(join('jones_fit_data', 'UVHWP_balance_sweep1.csv'))
    df_hh = pd.read_csv(join('jones_fit_data', 'UVHWP_balance_sweep_2.csv'))

    plt.errorbar(df_vv['C_UV_HWP position (deg)'], df_vv['C4 rate (#/s)'], yerr = df_vv['C4 rate SEM (#/s)'], fmt='o', label='VV')
    plt.errorbar(df_hh['C_UV_HWP position (deg)'], df_hh['C4 rate (#/s)'], yerr = df_hh['C4 rate SEM (#/s)'], fmt='o', label='HH')

    # fitting function #
    def sinsq2(x, a, b, c):
        return a* np.sin(np.deg2rad(2*x+b))**2 +c

    popt_vv, pcov_vv = curve_fit(sinsq2, df_vv['C_UV_HWP position (deg)'],df_vv['C4 rate (#/s)'], sigma=df_vv['C4 rate SEM (#/s)'])

    popt_hh, pcov_hh = curve_fit(sinsq2, df_hh['C_UV_HWP position (deg)'],df_hh['C4 rate (#/s)'], sigma=df_hh['C4 rate SEM (#/s)'])

    angles = np.linspace(min(df_hh['C_UV_HWP position (deg)']), max(df_hh['C_UV_HWP position (deg)']), 1000)

    plt.plot(angles, sinsq2(angles, *popt_vv), label='$%.3g \sin(2\\theta+%.3g)^2 + %.3g$'%(popt_vv[0], popt_vv[1], popt_vv[2]))
    plt.plot(angles, sinsq2(angles, *popt_hh), label='$%.3g \sin(2 \\theta + %.3g)^2 + %.3g$'%(popt_hh[0], popt_hh[1], popt_hh[2]))
    # plt.plot(angles, sinsq2(angles, *popt_vv), label='$%.3g \sin(2\\theta)^2 + %.3g$'%(popt_vv[0], popt_vv[1]))
    # plt.plot(angles, cossq2(angles, *popt_hh), label='$%.3g \cos(2 \\theta)^2 + %.3g$'%(popt_hh[0], popt_hh[1]))

    plt.title('Comparing max VV and max HH')
    plt.xlabel('Angles (deg)')
    plt.ylabel('C4 Coincidences')

    print('vv / hh', np.abs((popt_vv[0] + popt_vv[2]) / (popt_hh[0] + popt_hh[2])))
    s_vv = np.sqrt(np.diag(pcov_vv))[0]
    s_hh = np.sqrt(np.diag(pcov_hh))[0]
    print('unc',  np.sqrt((s_vv / popt_hh[0])**2 + (popt_vv[0] / (popt_hh[0])**2*s_hh)**2))
    print('middle point',df_vv['C4 rate (#/s)'][9] / df_hh['C4 rate (#/s)'][9])

    plt.legend()
    plt.savefig(join('hh_vv_data', 'max_vv_hh.pdf'))
    plt.show()

def det_qp_phi():
    '''Function to determine relation of QP rotation to theoretical phi'''

    # read in data #
    df = pd.read_csv('../../framework/qp_phi_sweep.csv')
    print('len of data', len(df[df['QP']<38.3]))

    # plot data #
    fig, ax = plt.subplots(2, 1, figsize=(10,7),sharex=True)

    # convert all angles to radians #
    QP_vals_a = np.deg2rad(df['QP'].to_numpy())
    phi_vals_a = np.deg2rad(df['phi'].to_numpy())
    phi_err_a= np.deg2rad(df['unc'].to_numpy())

    # restrict fit where phi goes from 0 to 2 pi
    phi_vals = phi_vals_a[phi_vals_a < 2*np.pi]
    QP_vals = QP_vals_a[phi_vals_a < 2*np.pi]
    phi_err = phi_err_a[phi_vals_a < 2*np.pi]


    ax[0].errorbar(QP_vals, phi_vals, yerr=phi_err, fmt='o')

    # fitting function #
    def fit(x, a, b, c, d, e, f, g,h, i, j):
        return a / np.cos(x) + b*x**8 + c*x**7 +d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j

    # do fit #
    popt, pcov = curve_fit(fit, QP_vals, phi_vals, sigma=phi_err)
    phi_ls = np.linspace(min(QP_vals), max(QP_vals), 1000)
    # ax[0].plot(phi_ls, fit(phi_ls, *popt))
    ax[0].plot(phi_ls, fit(phi_ls, *popt))
    print('$%.3g / \cos(\\theta) + %.3g \\theta^8 + %.3g \\theta^7 + %.3g \\theta^6 + %.3g \\theta^5 + %.3g \\theta^4 + %.3g \\theta^3 + %.3g \\theta^2 + %.3g \\theta + %.3g$'%(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9]))
    print(popt)

    ax[0].set_ylabel('Phi (rad)')
    ax[0].set_title('Phi vs QP rotation')
    # ax[0].legend()

    # print fit parameters #
    print('a', popt[0], np.sqrt(np.diag(pcov))[0])
    print('b', popt[1], np.sqrt(np.diag(pcov))[1])
    print('c', popt[2], np.sqrt(np.diag(pcov))[2])
    print('d', popt[3], np.sqrt(np.diag(pcov))[3])

    # compute chi2red #
    norm_resid = (phi_vals - fit(QP_vals, *popt)) / phi_err
    chi2 = np.sum(norm_resid**2)
    chi2red = chi2 / (len(QP_vals) - len(popt))
    print('chi2red', chi2red)
    # get num with 1 sigma of 0 #
    num_1sig = len(norm_resid[np.abs(norm_resid) < 1])
    print('percent within 1 sigma', num_1sig / len(norm_resid))

    # plot norm residuals #
    ax[1].errorbar(QP_vals, norm_resid, yerr=phi_err, fmt='o')
    ax[1].plot(phi_ls, np.zeros(len(phi_ls)), 'k--')

    ax[1].set_xlabel('QP rotation (rad)')
    ax[1].set_ylabel('Normalized Residuals')
    ax[1].set_title('Normalized residuals, $\chi^2_\\nu = %.3g$'%chi2red)

    plt.suptitle('Comparison of QP rotation to theoretical phi')
    plt.savefig(join('jones_fit_data', 'qp_phi.pdf'))
   

    plt.show()

def hh_vv_qp_sweep():
    '''Taking separate data for at HH and VV settings, varying QP rotation angle'''
    hh_df = pd.read_csv('../../framework/decomp_test/HH_20.csv')
    vv_df = pd.read_csv('../../framework/decomp_test/VV_20.csv')

    hh_counts = hh_df['C4 rate (#/s)'].to_numpy()
    hh_unc = hh_df['C4 rate SEM (#/s)'].to_numpy()
    vv_counts = vv_df['C4 rate (#/s)'].to_numpy()
    vv_unc = vv_df['C4 rate SEM (#/s)'].to_numpy()
    angles = hh_df['C_QP position (deg)'].to_numpy()

    # plot data #
    # plt.plot(angles, vv_counts / hh_counts, 'o', label='VV / HH')
    fig, ax = plt.subplots(1, 3, figsize=(15,5),sharex=True)
    ax[0].errorbar(angles, vv_counts, yerr = vv_unc, fmt='o',label='VV')
    ax[0].errorbar(angles, hh_counts, yerr = hh_unc, fmt='o', label='HH')
    ax[0].set_xlabel('QP rotation (deg)')
    ax[0].set_ylabel('Counts (#/s)')
    ax[0].set_title('Counts vs QP rotation')
    ax[0].legend()

    vv_hh_unc = np.sqrt((vv_unc / hh_counts)**2 + (hh_unc * vv_counts / hh_counts**2)**2)
    ax[1].errorbar(angles, vv_counts / hh_counts, yerr=vv_hh_unc, fmt='o')

    # fit data #
    def func(x, a, b):
        return a*x + b
    
    popt, pcov = curve_fit(func, angles, vv_counts / hh_counts, sigma=vv_hh_unc, absolute_sigma=True)
    angles_ls = np.linspace(min(angles), max(angles), 1000)
    ax[1].plot(angles_ls, func(angles_ls, *popt), label='%.3g \\theta_{QP} + %.3g'%(popt[0], popt[1]))
    ax[1].set_xlabel('QP rotation (deg)')
    ax[1].set_ylabel('VV / HH')
    ax[1].set_title('VV / HH vs QP rotation')
    ax[1].legend()

    print('a', popt[0], np.sqrt(np.diag(pcov))[0])
    print('b', popt[1], np.sqrt(np.diag(pcov))[1])


    # residuals #
    norm_resid = (vv_counts / hh_counts - func(angles, *popt)) / vv_hh_unc
    chi2 = np.sum(norm_resid**2)
    chi2red = chi2 / (len(angles) - len(popt))

    ax[2].errorbar(angles, norm_resid, yerr=np.ones_like(norm_resid), fmt='o')
    ax[2].plot(angles_ls, np.zeros(len(angles_ls)), 'k--')
    ax[2].set_ylabel('Normalized Residuals')
    ax[2].set_xlabel('QP rotation (deg)')
    ax[2].set_title('$\chi^2_\\nu = %.3g$'%chi2red)

    plt.suptitle('HH and VV counts vs QP rotation')
    plt.tight_layout()
    plt.savefig('jones_fit_data/hh_vv_qp_sweep_20.pdf')
    plt.show()

if __name__=='__main__':
    hh_vv_qp_sweep()