from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import uncertainties as unc

if __name__ == '__main__':
    DATE = '07162025'
    TRIAL = 4
    stateName = 'ha_negpi_3_vd'
    goalGamma = -np.pi/3
    fileName = f'QP_calibration/plotted_gamma_data_{DATE}_for_{stateName}_chi18_{TRIAL}'
    df = Manager.load_data(f'QP_calibration/all_gamma_data_{DATE}_for_{stateName}_chi18_{TRIAL}.csv')
    angles, gammas = df['QP'], df['gamma'].apply(unc.ufloat_fromstr)
   
    params = analysis.fit('line', angles, gammas)
    analysis.plot_func('line', params, angles, color='blue')
    analysis.plot_errorbar(angles, gammas, color='red', ms=0.1, fmt='o', label='Data')
    plt.legend()
    plt.xlabel('QP Angle (deg)')
    plt.ylabel('Gamma (rad)')
    plt.savefig(f'{fileName}.png', dpi=600)
    plt.show()

    x = analysis.find_value('line', params, goalGamma, angles)
    print(f'{goalGamma} at {x}')
    # df = Manager.load_data('all_gamma_chi_data_07022025_for_havd.csv')
    # fileName = f'plotted_gamma_chi_data_07022025_for_havd'
    # chis, gammas = df['chi'], df['gamma'].apply(unc.ufloat_fromstr)
   
    # params = analysis.fit('line', chis, gammas)
    # analysis.plot_func('line', params, chis, color='blue')
    # analysis.plot_errorbar(chis, gammas, color='red', ms=0.1, fmt='o', label='Data')
    # plt.legend()
    # plt.xlabel('Chi')
    # plt.ylabel('Gamma (rad)')
    # plt.savefig(f'{fileName}.png', dpi=600)
    # plt.show()
