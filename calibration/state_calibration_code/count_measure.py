from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp


if __name__ == '__main__':
    # initialize the manager
    m = Manager('../config.json')

    m.make_state('phi_plus')

    # parameters used for HD + e^-ipi/6 VA state
    m.log('Setting up state')
    m.B_C_HWP.goto(67.5)
    m.B_C_QWP.goto(45)
    m.C_QP.goto(-27.379)
    m.C_UV_HWP.goto(-115.56962464985094)

    m.log('Setting up measurement basis')
    m.meas_basis("VV")
    m.A_HWP.goto(-7.5)
    m.A_QWP.goto(-60)

    m.log('Checking count rates...')
    min_counts1 = m.take_data(6,5,'C4')

    m.log('Setting up measurement basis')
    m.meas_basis("HH")
    m.A_HWP.goto(-52.5)
    m.A_QWP.goto(30)

    m.log('Checking count rates...')
    min_counts2 = m.take_data(6,5,'C4')

    m.meas_basis("HD")
    hd_counts = m.take_data(6, 5, 'C4')
    m.meas_basis("VA")
    va_counts = m.take_data(6, 5, 'C4')
    m.meas_basis("HA")
    ha_counts = m.take_data(6, 5, 'C4')
    m.meas_basis("VD")
    vd_counts = m.take_data(6, 5, 'C4')
    
    # m.meas_basis("DA")
    # da_counts = m.take_data(6, 5, 'C4')
    # m.meas_basis("AD")
    # ad_counts = m.take_data(6, 5, 'C4')
    # m.meas_basis("DD")
    # dd_counts = m.take_data(6, 5, 'C4')
    # m.meas_basis("AA")
    # aa_counts = m.take_data(6, 5, 'C4')

    # tell the user what is up #Min Basis Counts: {min_counts} \n
    # print(f'Minimized basis count rates: {min_counts} \n HD Counts: {hd_counts} \n VA Counts: {va_counts} \n HA Counts: {ha_counts} \n VD Counts: {vd_counts}')
    print(f'Min counts 1: {min_counts1} \n Min counts 2: {min_counts2}')
    print(f' HD Counts: {hd_counts} \n VA Counts: {va_counts} \n HA Counts: {ha_counts} \n VD Counts: {vd_counts}')
    # print(f'AD Counts: {ad_counts} \n DA Counts: {da_counts} \n DD Counts: {dd_counts} \n AA Counts: {aa_counts}')

    print('Exiting...')
    m.shutdown()
    quit()