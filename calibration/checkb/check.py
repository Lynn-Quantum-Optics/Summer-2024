from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import os


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
    m.log('Loading VV state.')
    m.make_state('VV')
    m.log('Configuring Alice\'s measurement waveplates to measure V')
    m.log('AQWP -> 0')
    m.A_QWP.goto(0)
    m.log('AHWP -> 45')
    m.A_HWP.goto(45)

if __name__ == '__main__':
    # output folder
    TRIAL = 0

    # make output folder
    outdir = f'./sweeps{TRIAL}'
    if os.isdir(outdir):
        print('Output folder already exists')
        return 0
    else:
        os.mkdir(outdir)

    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log('CHECKPOINT B: Checking calibrations on B_HWP, B_QWP, B_C_HWP, and B_C_QWP.')

    # do a mini-sweep for each component. reset everything in between
    reset(m)
    BHWP_ext = mini_sweep(m, 'B_HWP', f'{outdir}/BHWP.csv', f'{outdir}/BHWP.png')
    reset(m)
    BQWP_ext = mini_sweep(m, 'B_QWP', f'{outdir}/BQWP.csv', f'{outdir}/BQWP.png')
    reset(m)
    BCHWP_ext = mini_sweep(m, 'B_C_HWP', f'{outdir}/BCHWP.csv', f'{outdir}/BCHWP.png')
    reset(m)
    BCQWP_ext = mini_sweep(m, 'B_C_QWP', f'{outdir}/BCQWP.csv', f'{outdir}/BCQWP.png')
    m.shutdown()

    # print out the updates
    print(f'BHWP extrema: {BHWP_ext}')
    print(f'BHWP Update: {m.B_HWP.offset:.3f} + {BHWP_ext.n:.3f} -> {m.B_HWP.offset + BHWP_ext.n:.3f}')
    print(f'BQWP extrema: {BQWP_ext}')
    print(f'BQWP Update: {m.B_QWP.offset:.3f} + {BQWP_ext.n:.3f} -> {m.B_QWP.offset + BQWP_ext.n:.3f}')
    print(f'BCHWP extrema: {BCHWP_ext}')
    print(f'BCHWP Update: {m.B_C_HWP.offset:.3f} + {BCWP_ext.n:.3f} -> {m.B_C_HWP.offset + BCHWP_ext.n:.3f}')
    print(f'BCQWP extrema: {BCQWP_ext}')
    print(f'BCQWP Update: {m.B_C_QWP.offset:.3f} + {BCQWP_ext.n:.3f} -> {m.B_C_QWP.offset + BCQWP_ext.n:.3f}')
