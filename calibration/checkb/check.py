from typing import List
from lab_framework import Manager, analysis
import matplotlib.pyplot as plt
import os


def mini_sweep(m:Manager, component:str, sweep_params:List[int], data_out_file:str, plot_out_file:str):
    # sweep -4 to 4 degrees
    angles, rates = m.sweep(component, *sweep_params)
    
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
    m.log('Loading presets for VV state.')
    m.make_state('VV')
    m.log('Configuring Alice\'s measurement waveplates to measure V')
    m.log(f'AQWP -> {m.A_QWP.goto(0)}')
    m.log(f'AHWP -> {m.A_HWP.goto(45)}')
    m.log('Configuring Bob\'s Waveplates to Minimize C4 Coincidences')
    m.log(f'BQWP -> {m.B_QWP.goto(0)}')
    m.log(f'BHWP -> {m.B_HWP.goto(0)}')
    m.log(f'BCHWP -> {m.B_C_HWP.goto(0)}')

if __name__ == '__main__':
    # output folder
    # TRIAL = 0
    # SWEEP_PARAMS = (-4, 4, 15, 5, 1)
    TRIAL = 2
    SWEEP_PARAMS = (-6, 6, 20, 5, 3)

    # make output folder
    outdir = f'./sweeps{TRIAL}'
    if os.path.isdir(outdir):
        print('Output folder already exists')
        quit()
    else:
        os.mkdir(outdir)

    # initialize the manager
    m = Manager(config='../config.json')

    # log session info
    m.log('CHECKPOINT B: Checking calibrations on B_HWP, B_QWP, and B_C_HWP.')

    # do a mini-sweep for each component. reset everything in between
    reset(m)
    BHWP_ext = mini_sweep(m, 'B_HWP', SWEEP_PARAMS, f'{outdir}/BHWP.csv', f'{outdir}/BHWP.png')
    BHWP_off = m.B_HWP.offset
    reset(m)
    BQWP_ext = mini_sweep(m, 'B_QWP', SWEEP_PARAMS, f'{outdir}/BQWP.csv', f'{outdir}/BQWP.png')
    BQWP_off = m.B_QWP.offset
    reset(m)
    BCHWP_ext = mini_sweep(m, 'B_C_HWP', SWEEP_PARAMS, f'{outdir}/BCHWP.csv', f'{outdir}/BCHWP.png')
    BCHWP_off = m.B_C_HWP.offset
    # reset(m)
    BCQWP_ext = mini_sweep(m, 'B_C_QWP', SWEEP_PARAMS, f'{outdir}/BCQWP.csv', f'{outdir}/BCQWP.png')
    m.shutdown()

    # print out the updates
    print(f'BHWP extrema: {BHWP_ext}')
    print(f'BHWP Update: {BHWP_off:.3f} + {BHWP_ext.n:.3f} -> {BHWP_off + BHWP_ext.n:.3f}')
    print(f'BQWP extrema: {BQWP_ext}')
    print(f'BQWP Update: {BQWP_off:.3f} + {BQWP_ext.n:.3f} -> {BQWP_off + BQWP_ext.n:.3f}')
    print(f'BCHWP extrema: {BCHWP_ext}')
    print(f'BCHWP Update: {BCHWP_off:.3f} + {BCHWP_ext.n:.3f} -> {BCHWP_off + BCHWP_ext.n:.3f}')
    print(f'BCQWP extrema: {BCQWP_ext}')
    print(f'BCQWP Update: {m.B_C_QWP.offset:.3f} + {BCQWP_ext.n:.3f} -> {m.B_C_QWP.offset + BCQWP_ext.n:.3f}')
