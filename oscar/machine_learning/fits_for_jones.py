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
    fig, ax = plt.subplots(2, 2, figsize=(10,7))

    # convert all angles to radians #
    QP_vals_a = np.deg2rad(df['QP'].to_numpy())
    phi_vals_a = np.deg2rad(df['phi'].to_numpy())
    phi_err_a= np.deg2rad(df['unc'].to_numpy())

    # restrict fit where phi goes from 0 to 2 pi plus offset of whatver 0 is
    phi_vals = phi_vals_a[phi_vals_a < (2*np.pi + phi_vals_a[0])]
    QP_vals = QP_vals_a[phi_vals_a < (2*np.pi + phi_vals_a[0])]
    phi_err = phi_err_a[phi_vals_a < (2*np.pi + phi_vals_a[0])]

    ax[0,0].errorbar(QP_vals, phi_vals, yerr=phi_err, fmt='o')

    # fitting function #
    def fit(x, a, b, c, d, e, f, g,h, i, j):
        return a / np.cos(x) + b*x**8 + c*x**7 +d*x**6 + e*x**5 + f*x**4 + g*x**3 + h*x**2 + i*x + j

    # do fit #
    popt, pcov = curve_fit(fit, QP_vals, phi_vals, sigma=phi_err)
    phi_ls = np.linspace(min(QP_vals), max(QP_vals), 1000)
    # ax[0].plot(phi_ls, fit(phi_ls, *popt))
    ax[0,0].plot(phi_ls, fit(phi_ls, *popt), 'r--')
    print('$%.3g / \cos(\\theta) + %.3g \\theta^8 + %.3g \\theta^7 + %.3g \\theta^6 + %.3g \\theta^5 + %.3g \\theta^4 + %.3g \\theta^3 + %.3g \\theta^2 + %.3g \\theta + %.3g$'%(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9]))
    print('pos params', popt)

    ax[0,0].set_ylabel('$\phi$ (rad)')
    # ax[0,0].set_title('$\phi$ vs QP rotation')
    # ax[0].legend()

    # compute chi2red #
    norm_resid = (phi_vals - fit(QP_vals, *popt)) / phi_err
    chi2 = np.sum(norm_resid**2)
    chi2red = chi2 / (len(QP_vals) - len(popt))
    print('chi2red', chi2red)
    # get num with 1 sigma of 0 #
    num_1sig = len(norm_resid[np.abs(norm_resid) < 1])
    print('percent within 1 sigma', num_1sig / len(norm_resid))

    # plot norm residuals #
    ax[0,1].errorbar(QP_vals, norm_resid, yerr=phi_err, fmt='o')
    ax[0,1].plot(phi_ls, np.zeros(len(phi_ls)), 'k--')

    # ax[0,1].set_xlabel('QP rotation (rad)')
    ax[0,1].set_ylabel('Normalized Residuals')
    ax[0,1].set_title('$\chi^2_\\nu = %.3g$'%chi2red)

    # plot negaitve QP rot #
    df_neg = pd.read_csv('../../framework/qp_phi_sweep_neg.csv')
    QP_vals_neg_a = np.deg2rad(df_neg['QP'].to_numpy()) 
    phi_vals_neg = np.deg2rad(df_neg['phi'].to_numpy()) % (2*np.pi)
    phi_err_neg = np.deg2rad(df_neg['unc'].to_numpy())

    # add first 5 data points of positive data to neg
    # QP_vals_neg_a = np.append(QP_vals_neg_a, QP_vals_a[:5])
    # phi_vals_neg = np.append(phi_vals_neg, phi_vals_a[:5])
    # phi_err_neg = np.append(phi_err_neg, phi_err_a[:5])

    # restrict QP_vals
    QP_cutoff = -.6363
    QP_vals_neg = QP_vals_neg_a[QP_vals_neg_a >= QP_cutoff]
    phi_vals_neg = phi_vals_neg[QP_vals_neg_a >= QP_cutoff]
    phi_err_neg = phi_err_neg[QP_vals_neg_a >= QP_cutoff]

    # QP_vals_neg = QP_vals_neg_a

    ax[1,0].errorbar(QP_vals_neg, phi_vals_neg, yerr=phi_err_neg, fmt='o')

    # do fit #
    popt_neg, pcov_neg = curve_fit(fit, QP_vals_neg, phi_vals_neg, sigma=phi_err_neg)
    phi_ls_neg = np.linspace(min(QP_vals_neg), max(QP_vals_neg), 1000)
    # ax[0].plot(phi_ls, fit(phi_ls, *popt))
    ax[1,0].plot(phi_ls_neg, fit(phi_ls_neg, *popt_neg), 'r--')
    print('$%.3g / \cos(\\theta) + %.3g \\theta^8 + %.3g \\theta^7 + %.3g \\theta^6 + %.3g \\theta^5 + %.3g \\theta^4 + %.3g \\theta^3 + %.3g \\theta^2 + %.3g \\theta + %.3g$'%(popt_neg[0], popt_neg[1], popt_neg[2], popt_neg[3], popt_neg[4], popt_neg[5], popt_neg[6], popt_neg[7], popt_neg[8], popt_neg[9]))
    print('neg params', popt_neg)

    ax[1,0].set_ylabel('$\phi$ (rad)')
    ax[1,0].set_xlabel('$\\theta_{QP}$ (rad)')

    # compute chi2red #
    norm_resid_neg = (phi_vals_neg - fit(QP_vals_neg, *popt_neg)) / phi_err_neg
    chi2_neg = np.sum(norm_resid_neg**2)
    chi2red_neg = chi2_neg / (len(QP_vals_neg) - len(popt_neg))
    print('chi2red', chi2red_neg)
    # get num with 1 sigma of 0 #
    num_1sig_neg = len(norm_resid_neg[np.abs(norm_resid_neg) < 1])
    print('percent within 1 sigma', num_1sig_neg / len(norm_resid_neg))

    # plot norm residuals #
    ax[1,1].errorbar(QP_vals_neg, norm_resid_neg, yerr=phi_err_neg, fmt='o')
    ax[1,1].plot(phi_ls_neg, np.zeros(len(phi_ls_neg)), 'k--')
    ax[1,1].set_xlabel('$\\theta_{QP}$ (rad)')
    ax[1,1].set_ylabel('Normalized Residuals')
    ax[1,1].set_title('$\chi^2_\\nu = %.3g$'%chi2red_neg)

    plt.suptitle('Comparison of QP rotation $\\theta_{QP}$ to theoretical $\phi$')
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
    fig, ax = plt.subplots(2, 3, figsize=(15,5))
    ax[0,0].errorbar(angles, vv_counts, yerr = vv_unc, fmt='o',label='VV')
    ax[0,0].errorbar(angles, hh_counts, yerr = hh_unc, fmt='o', label='HH')
    ax[0,0].set_ylabel('Counts (#/s)')
    ax[0,0].set_title('Counts vs QP rotation')
    ax[0,0].legend()

    vv_hh_f_unc = np.sqrt(( vv_unc / (hh_counts+vv_counts)*(vv_counts * vv_unc) / (hh_counts + vv_counts)**2)**2 + (hh_unc * vv_counts / (hh_counts + vv_counts)**2)**2)

    ax[0,1].errorbar(angles, vv_counts / (hh_counts+vv_counts), yerr=vv_hh_f_unc, fmt='o')

    # fit data #
    def func(x, a, b, c, d, e):
        return a*x**4 + b*x*3 + c*x**2 + d*x +e
    
    popt, pcov = curve_fit(func, angles, vv_counts / (hh_counts + vv_counts), sigma=vv_hh_f_unc, absolute_sigma=True)
    angles_ls = np.linspace(min(angles), max(angles), 1000)
    ax[0,1].plot(angles_ls, func(angles_ls, *popt), 'r--')
    ax[0,1].set_ylabel('$\\frac{VV}{HH + VV}$')
    ax[0,1].set_title('$\\frac{VV}{HH + VV}$ vs QP rotation')

    print('pos params', popt)

    # residuals #
    norm_resid = (vv_counts / (hh_counts+vv_counts) - func(angles, *popt)) / vv_hh_f_unc
    chi2 = np.sum(norm_resid**2)
    chi2red = chi2 / (len(angles) - len(popt))

    ax[0,2].errorbar(angles, norm_resid, yerr=np.ones_like(norm_resid), fmt='o')
    ax[0,2].plot(angles_ls, np.zeros(len(angles_ls)), 'k--')
    ax[0,2].set_ylabel('Normalized Residuals')
    ax[0,2].set_title('$\chi^2_\\nu = %.3g$'%chi2red)

    # do for negtive QP rot #
    hh_hh_df = pd.read_csv('../../framework/decomp_test/HH_HH_20_-38_0.csv')
    # hh_hv_df = pd.read_csv('../../framework/decomp_test/HH_HV_5_-45_0.csv')
    # hh_vh_df = pd.read_csv('../../framework/decomp_test/HH_VH_5_-45_0.csv')
    # hh_vv_df = pd.read_csv('../../framework/decomp_test/HH_VV_5_-45_0.csv')
    # vv_hh_df = pd.read_csv('../../framework/decomp_test/VV_HH_5_-45_0.csv')
    # vv_hv_df = pd.read_csv('../../framework/decomp_test/VV_HV_5_-45_0.csv')
    # vv_vh_df = pd.read_csv('../../framework/decomp_test/VV_VH_5_-45_0.csv')
    vv_vv_df = pd.read_csv('../../framework/decomp_test/VV_VV_20_-38_0.csv')

    hh_hh_counts = hh_hh_df['C4 rate (#/s)'].to_numpy()
    hh_hh_unc = hh_hh_df['C4 rate SEM (#/s)'].to_numpy()
    # hh_hv_counts = hh_hv_df['C4 rate (#/s)'].to_numpy()
    # hh_hv_unc = hh_hv_df['C4 rate SEM (#/s)'].to_numpy()
    # hh_vh_counts = hh_vh_df['C4 rate (#/s)'].to_numpy()
    # hh_vh_unc = hh_vh_df['C4 rate SEM (#/s)'].to_numpy()
    # hh_vv_counts = hh_vv_df['C4 rate (#/s)'].to_numpy()
    # hh_vv_unc = hh_vv_df['C4 rate SEM (#/s)'].to_numpy()
    # vv_hh_counts = vv_hh_df['C4 rate (#/s)'].to_numpy()
    # vv_hh_unc = vv_hh_df['C4 rate SEM (#/s)'].to_numpy()
    # vv_hv_counts = vv_hv_df['C4 rate (#/s)'].to_numpy()
    # vv_hv_unc = vv_hv_df['C4 rate SEM (#/s)'].to_numpy()
    # vv_vh_counts = vv_vh_df['C4 rate (#/s)'].to_numpy()
    # vv_vh_unc = vv_vh_df['C4 rate SEM (#/s)'].to_numpy()
    vv_vv_counts = vv_vv_df['C4 rate (#/s)'].to_numpy()
    vv_vv_unc = vv_vv_df['C4 rate SEM (#/s)'].to_numpy()

    angles_o = hh_hh_df['C_QP position (deg)'].to_numpy()
    angles = angles_o[(angles_o >= 323.5427) & (angles_o <=360)]
    hh_hh_counts = hh_hh_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    hh_hh_unc = hh_hh_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_hv_counts = hh_hv_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_hv_unc = hh_hv_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_vh_counts = hh_vh_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_vh_unc = hh_vh_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_vv_counts = hh_vv_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # hh_vv_unc = hh_vv_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_hh_counts = vv_hh_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_hh_unc = vv_hh_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_hv_counts = vv_hv_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_hv_unc = vv_hv_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_vh_counts = vv_vh_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    # vv_vh_unc = vv_vh_unc[(angles_o >= 323.5427) & (angles_o <=360)]
    vv_vv_counts = vv_vv_counts[(angles_o >= 323.5427) & (angles_o <=360)]
    vv_vv_unc = vv_vv_unc[(angles_o >= 323.5427) & (angles_o <=360)]

    # plot data #
    ax[1,0].errorbar(angles, vv_vv_counts, yerr = vv_vv_unc, fmt='o',label='VV')
    ax[1,0].errorbar(angles, hh_hh_counts, yerr = hh_hh_unc, fmt='o', label='HH')
    # ax[1,0].set_xlim(323.5427, 360)
    ax[1,0].set_xlabel('QP rotation (deg)')
    ax[1,0].set_ylabel('Counts (#/s)')
    ax[1,0].legend()

    hh_vv_f_unc = np.sqrt(( hh_hh_unc / (hh_hh_counts+vv_vv_counts)*(vv_vv_counts * hh_hh_unc) / (hh_hh_counts + vv_vv_counts)**2)**2 + (vv_vv_unc * hh_hh_counts / (hh_hh_counts + vv_vv_counts)**2)**2)
    
    ax[1,1].errorbar(angles, vv_vv_counts / (hh_hh_counts+vv_vv_counts), yerr=hh_vv_f_unc, fmt='o')

    print('fit angles', angles)
    # fit #
    popt, pcov = curve_fit(func, angles, vv_vv_counts / (hh_hh_counts+vv_vv_counts), sigma=hh_vv_f_unc, absolute_sigma=True)

    angles_ls = np.linspace(323.5427, 360, 1000)
    ax[1,1].plot(angles_ls, func(angles_ls, *popt),'r--')
    # ax[1,1].set_xlim(323.5427, 360)
    ax[1,1].set_xlabel('QP rotation (deg)')
    ax[1,1].set_ylabel('$\\frac{VV}{HH + VV}$')

    # print(func(335, *popt))

    print('neg params', popt)

    # get chi2_red #
    norm_resid = (vv_vv_counts / (hh_hh_counts+vv_vv_counts) - func(angles, *popt)) / hh_vv_f_unc
    chi2red = np.sum(norm_resid**2) / (len(angles) - len(popt))

    ax[1,2].errorbar(angles, norm_resid, yerr=np.ones_like(norm_resid), fmt='o')
    ax[1,2].plot(angles_ls, np.zeros(len(angles_ls)), 'k--')
    ax[1,2].set_title('$\chi^2_\\nu = %.3g$'%chi2red)
    # ax[1,2].set_xlim(323.5427, 360)
    ax[1,2].set_ylabel('Normalized Residuals')
    ax[1,2].set_xlabel('QP rotation (deg)')
    
    plt.suptitle('HH and VV counts vs QP rotation')
    plt.tight_layout()
    plt.savefig('jones_fit_data/hh_vv_qp_sweep_20.pdf')
    plt.show()

    # plot individual components #
    # plt.figure(figsize=(10,7))
    # plt.errorbar(angles, hh_hv_counts, yerr = hh_hv_unc, fmt='o',label='HH, HV')
    # plt.errorbar(angles, hh_vh_counts, yerr = hh_hh_unc, fmt='o', label='HH, VH')
    # plt.errorbar(angles, hh_vv_counts, yerr = hh_vv_unc, fmt='o', label='HH, VV')
    # plt.errorbar(angles, vv_hh_counts, yerr = vv_hh_unc, fmt='o',label='VV, HH')
    # plt.errorbar(angles, vv_hv_counts, yerr = vv_hv_unc, fmt='o', label='VV, HV')
    # plt.errorbar(angles, vv_vh_counts, yerr = vv_vh_unc, fmt='o', label='VV, VH')
    # plt.legend()
    # # plt.xlim(323.5427, 360)
    # plt.xlabel('QP rotation (deg)')
    # plt.ylabel('Counts (#/s)')
    # plt.title('Other bases for HH, VV states')
    # plt.savefig('jones_fit_data/qp_extra_bases.pdf')
    # # plt.show()

if __name__=='__main__':
    hh_vv_qp_sweep()
    # det_qp_phi()