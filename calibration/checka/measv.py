from lab_framework import Manager, analysis
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    # initialize manager
    m = Manager('../config.json')
    
    # make the VX state
    m.make_state('VX')

    # move alice's QWP to zero
    m.log('Moving Alice\'s QWP to calibrated zero')
    m.A_QWP.goto(0)

    # sweep alice's HWP to measure in the V basis
    angles, rates = m.sweep('A_HWP', 40, 50, 11, 5, 1)
    m.output_data('measv_ahwp_sweep.csv')
    m.shutdown()
    
    # do a fit
    params = analysis.fit('quadratic', angles, rates)
    ext = params[0]
    '''
    df = Manager.load_data('measv_ahwp_sweep.csv')
    angles, rates = df['A_HWP'], df['C4']

    # do a fit
    # params = analysis.fit('quadratic', angles, rates)
    # ext = params[0]

    # plot those suckers
    analysis.plot_errorbar(angles, rates, fmt='ro')
    # analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed')
    plt.xlabel('Alice\'s HWP Angle (degrees)')
    plt.ylabel('C4 Count Rates (#/s)')
    # plt.title(f'Extrema at {ext} degrees')
    plt.savefig('measv_ahwp_sweep_plot.png')