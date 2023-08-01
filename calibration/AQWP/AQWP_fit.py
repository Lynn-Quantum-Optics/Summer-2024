from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

def fit_func(theta, phi, alpha, N, C):
    return N*(cos(deg2rad(alpha))**2 - 0.5*cos(2*deg2rad(alpha))*sin(2*deg2rad(theta-phi))**2) + C

if __name__ == '__main__':
    
    TRIAL = 2
    SWEEP_PARAMS = [-15, -3, 30, 5, 3]
    UVHWP_ANGLE = 0

    '''
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'AQWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}; UVHWP ANGLE = {UVHWP_ANGLE}')

    # configure the UVHWP to produce _something_
    m.C_UV_HWP.goto(UVHWP_ANGLE)

    # sweep alice's quarter waveplate
    m.sweep('A_QWP', *SWEEP_PARAMS)

    # get the output
    df = m.output_data(f'AQWP_sweep{TRIAL}.csv')
    m.shutdown()
    '''

    df = Manager.load_data('AQWP_sweep1.csv')
    
    # fit the function
    params = analysis.fit(fit_func, df['A_QWP'], df['C4'], p0=[0, 90.1518, 2423, -46], bounds=[[-180, -180, -inf, -inf], [180, 180, inf, inf]], maxfev=1000)
    # params = analysis.fit('quadratic', df['A_QWP'], df['C4'])

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # plotting
    # analysis.plot_func('quadratic', params, df['A_QWP'], color='b', linestyle='dashed', label=f'Fit Function', alpha=0.3)
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(1,1,1)
    analysis.plot_func(fit_func, params, df['A_QWP'], color='b', linestyle='dashed', label=f'${params[2].n:.3f}[\\cos^2({params[1].n:.3f})+\\cos(2\\cdot{params[1].n:.3f})\\sin^2(2(\\theta{params[0].n:.3f}))/2]+{params[3].n:.3f}$', alpha=0.3)
    analysis.plot_errorbar(df['A_QWP'], df['C4'], ms=0.1, fmt='ro', capsize=2, label='Data')
    plt.xlabel('Alice\'s QWP Angle (degrees)')
    plt.ylabel('Count Rate (#/s)')
    plt.legend()
    # plt.title(f'Fit=${params[1].n:.3f}(x-{params[0].n:.3f})^2 + {params[2].n:.3f}$')
    plt.title(f'Alice QWP Sweep')
    plt.savefig(f'AQWP_fit.png', dpi=600)
    plt.show()
