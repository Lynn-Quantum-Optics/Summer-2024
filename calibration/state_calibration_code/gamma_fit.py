from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp

if __name__ == '__main__':
    DATE = '06052025'
    TRIAL = 1
    goalGamma = -1*np.pi/3
    fileName = f'plotted_gamma_data_{DATE}_{TRIAL}'
    df = Manager.load_data(f'all_gamma_data_{DATE}_{TRIAL}.csv')
    angles, gammas = df['QP'], df['gamma']
   
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