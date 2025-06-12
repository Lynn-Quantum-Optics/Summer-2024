from lab_framework import Manager, analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp


if __name__ == '__main__':
    # initialize the manager
    m = Manager('../config.json')

    state = 'hr_negpi_6_vl'

    m.make_state('phi_plus')

    if state == 'hd_negpi_3_va':
        # parameters used for HD + e^-ipi/3 VA state
        m.log('Setting up state')
        m.B_C_HWP.goto(67.5)
        m.B_C_QWP.goto(45)
        m.C_QP.goto(-26.98906575253135) #-26.773838565224096
        m.C_UV_HWP.goto(-114.90445548609685)

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

        # tell the user the counts
        print(f'Min counts 1: {min_counts1} \n Min counts 2: {min_counts2}')
        print(f' HD Counts: {hd_counts} \n VA Counts: {va_counts} \n HA Counts: {ha_counts} \n VD Counts: {vd_counts}')
    
    if state == 'hr_negpi_6_vl':
        # parameters used for HD + e^-ipi/3 VA state
        m.log('Setting up state')
        m.B_C_HWP.goto(0)
        m.B_C_QWP.goto(-45)
        m.C_QP.goto(-12.11346435546875) #-26.773838565224096
        m.C_UV_HWP.goto(-111.29939852262797)

        m.log('Setting up measurement basis')
        m.meas_basis("VV")
        m.A_HWP.goto(-15)
        m.A_QWP.goto(-75)

        m.log('Checking count rates...')
        min_counts1 = m.take_data(6,5,'C4')

        m.log('Setting up measurement basis')
        m.meas_basis("HH")
        m.A_HWP.goto(-60)
        m.A_QWP.goto(15)

        m.log('Checking count rates...')
        min_counts2 = m.take_data(6,5,'C4')

        m.meas_basis("HR")
        hr_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("VL")
        vl_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("HL")
        hl_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("VR")
        vr_counts = m.take_data(6, 5, 'C4')

        # tell the user the counts
        print(f'Min counts 1: {min_counts1} \n Min counts 2: {min_counts2}')
        print(f' HR Counts: {hr_counts} \n VL Counts: {vl_counts} \n HL Counts: {hl_counts} \n VR Counts: {vr_counts}')

    if state == 'phi_plus':
        m.meas_basis("HH")
        hh_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("VV")
        vv_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("HV")
        hv_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("VH")
        vh_counts = m.take_data(6, 5, 'C4')
        
        m.meas_basis("DA")
        da_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("AD")
        ad_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("DD")
        dd_counts = m.take_data(6, 5, 'C4')
        m.meas_basis("AA")
        aa_counts = m.take_data(6, 5, 'C4')

        # tell the user the counts
        print(f'HH Counts: {hh_counts} \n VV Counts: {vv_counts} \n VH Counts: {vh_counts} \n HV Counts: {hv_counts}')
        print(f'AD Counts: {ad_counts} \n DA Counts: {da_counts} \n DD Counts: {dd_counts} \n AA Counts: {aa_counts}')


    print('Exiting...')
    m.shutdown()
    quit()