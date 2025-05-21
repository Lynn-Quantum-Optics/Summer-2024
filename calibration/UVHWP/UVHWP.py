from lab_framework import Manager, analysis
from numpy import sin, cos, deg2rad, inf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRIAL = 7
    SWEEP_PARAMS = [-10, 10, 20, 5, 1]
    
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log(f'UVHWP.py TRIAL # {TRIAL}; SWEEP PARAMS = {SWEEP_PARAMS}')

    # configure the UVHWP to produce _something_
    m.log('Sending Alice\'s QWP to calibrated zero')
    #m.A_QWP.goto(0) If you have to recalibrate measurement waveplates uncomment this line
    m.meas_basis('HH')
    # comment line 19 out if you have to recalibrate measurement waveplates
    m.C_QP.goto(0)
    m.C_PCC.goto(0)
    # just telling the QP and PPC to go to 0 if they are in 


    # sweep alice's quarter waveplate
    m.log('Sweeping UVHWP')
    m.sweep('C_UV_HWP', *SWEEP_PARAMS)

    # get the output
    df = m.output_data(f'UVHWP_sweep{TRIAL}.csv')
    m.shutdown()
    
    df = Manager.load_data(f'UVHWP_sweep{TRIAL}.csv')
    # '''
    # fit the function
    params = analysis.fit('quadratic', df['C_UV_HWP'], df['C4'])

    # print fitted parameters
    print(f'Fit parameters = {params}')

    # setup plot
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    
    # plotting
    analysis.plot_func('quadratic', params, df['C_UV_HWP'], color='b', linestyle='dashed', label=f'${params[1].n:.3f}(x-({params[0].n:.3f}))^2 + {params[2].n:.3f}$', alpha=0.3)
    analysis.plot_errorbar(df['C_UV_HWP'], df['C4'], ms=0.1, fmt='ro', capsize=2, label='Data')

    # labels and such
    plt.xlabel('UVHWP Angle (deg)')
    plt.ylabel('Count Rate (#/s)')
    plt.title('UV HWP Sweep')
    plt.legend()

    # save and show
    plt.savefig(f'UVHWP_sweep{TRIAL}.png', dpi=600)
    plt.show()
