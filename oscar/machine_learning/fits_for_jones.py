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

if __name__=='__main__':
    det_qp_phi()