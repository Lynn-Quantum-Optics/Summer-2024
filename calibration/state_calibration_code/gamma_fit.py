from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import uncertainties as unc

if __name__ == '__main__':
    DATE = '06112025'
    TRIAL = 1
    stateName = 'hr_negpi_6_vl'
    goalGamma = -1*np.pi/6
    fileName = f'plotted_gamma_data_{DATE}_{TRIAL}_for_{stateName}'
    df = Manager.load_data(f'all_gamma_data_{DATE}_{TRIAL}_for_{stateName}.csv')
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