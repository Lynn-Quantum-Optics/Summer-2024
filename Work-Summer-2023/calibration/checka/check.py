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
    m.log('Loading VV state.')
    m.make_state('VV')
    m.log('Zeroing Alice\'s measurement waveplates')
    m.log(f'AQWP -> {m.A_QWP.goto(0)}')
    m.log(f'AHWP -> {m.A_HWP.goto(0)}')

if __name__ == '__main__':
    # output folder
    TRIAL = 2
    SWEEP_PARAMS = (-4, 4, 20, 5, 3)

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
    m.log('CHECKPOINT A: Calibrated Alice\'S measurement waveplates and the UVHWP')

    # do a mini-sweep for each component. reset everything in between
    reset(m)
    UVHWP_ext = mini_sweep(m, 'C_UV_HWP', SWEEP_PARAMS, f'{outdir}/UVHWP.csv', f'{outdir}/UVHWP.png')
    UVHWP_off = m.C_UV_HWP.offset
    reset(m)
    AHWP_ext = mini_sweep(m, 'A_HWP', SWEEP_PARAMS, f'{outdir}/AHWP.csv', f'{outdir}/AHWP.png')
    AHWP_off = m.A_HWP.offset
    reset(m)
    AQWP_ext = mini_sweep(m, 'A_QWP', SWEEP_PARAMS, f'{outdir}/AQWP.csv', f'{outdir}/AQWP.png')
    AQWP_off = m.A_QWP.offset
    m.shutdown()
    
    print(f'UVHWP extrema: {UVHWP_ext}')
    print(f'UVHWP Update: {UVHWP_off:.3f} + {UVHWP_ext.n:.3f} -> {UVHWP_off + UVHWP_ext.n:.3f}')
    print(f'AHWP extrema: {AHWP_ext}')
    print(f'AHWP Update: {AHWP_off:.3f} + {AHWP_ext.n:.3f} -> {AHWP_off + AHWP_ext.n:.3f}')
    print(f'AQWP extrema: {AQWP_ext}')
    print(f'AQWP Update: {AQWP_off:.3f} + {AQWP_ext.n:.3f} -> {AQWP_off + AQWP_ext.n:.3f}')