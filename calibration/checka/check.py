from lab_framework import Manager, analysis
import matplotlib.pyplot as plt


def mini_sweep(m:Manager, component:str, data_out_file:str, plot_out_file:str):
    # sweep -4 to 4 degrees
    angles, rates = m.sweep(component, -4, 4, 16, 5, 3)
    
    # output the data
    m.output_data(data_out_file)

    # do a quadratic fit
    params = analysis.fit('quadratic', angles, rates)
    ext = params[0] # extremal value for x

    # make a plot!
    analysis.plot_errorbar(angles, rates, fmt='ro')
    analysis.plot_func('quadratic', params, angles, color='b', linestyle='dashed')
    plt.xlabel(f'{component} Angle (degrees)')
    plt.ylabel(f'C4 Count Rates (#/s)')
    plt.title(f'Extrema at {ext} degrees')
    plt.savefig(plot_out_file)
    plt.cla()

    return ext

def reset(m:Manager):
    m.log('Resetting...')
    m.log('Loading VX state.')
    m.make_state('VX')
    m.log('Zeroing Alice\'s measurement waveplates')
    m.A_QWP.goto(0)
    m.A_HWP.goto(0)

if __name__ == '__main__':
    
    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log('CHECKPOINT A: Calibrated Alice\'S measurement waveplates and the UVHWP')

    # do a mini-sweep for each component. reset everything in between
    reset(m)
    mini_sweep(m, 'C_UV_HWP', 'UVHWP.csv', 'UVHWP.png')
    reset(m)
    mini_sweep(m, 'A_HWP', 'AHWP.csv', 'AHWP.png')
    reset(m)
    mini_sweep(m, 'A_QWP', 'AQWP.csv', 'AQWP.png')
    m.shutdown()
