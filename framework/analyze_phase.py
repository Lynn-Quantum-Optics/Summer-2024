'''
Author: Ria Haapala
Last Update: 7/2/2025

This file plots the phase drift of a state over all chi values given the density matrices generated in data collection.

Note: this file assumes the phase is present in the density matrix with the form a = cos(phase) + i*sin(phase) such 
that phase = arctan2(Im(a)/Re(a)). If this is not the case for the state you are working with, duplicate this file 
and modify it as necessary.
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRIAL = 1
    chis = np.linspace(0.001, np.pi/2, 6)
    directory = 'stu_havd_trial0'
    all_phases = False

    fig, ax = plt.subplots()

    # get each phase
    for chi in chis:
        filename =f"rho_('E0', (45.0, {np.rad2deg(chi)}))_{TRIAL}.npy"
        data = np.load(f'{directory}/{filename}', allow_pickle=True)
        rho = data[0]

        if all_phases == True:
            fig_name = 'phase_drift_data_all_entries.png'
            phase_entry1 = rho[0][2]
            phase_entry2 = rho[0][3]
            phase_entry3 = rho[1][2]
            phase_entry4 = rho[1][3]
            phase1 = np.arctan(np.imag(phase_entry1)/np.real(phase_entry1))
            phase2 = np.arctan(np.imag(phase_entry2)/np.real(phase_entry2))
            phase3 = np.arctan(np.imag(phase_entry3)/np.real(phase_entry3))
            phase4 = np.arctan(np.imag(phase_entry4)/np.real(phase_entry4))

            phase = np.mean([phase1, phase2, phase3, phase4])
        else:
            fig_name = 'phase_drift_data_one_entry.png'
            phase_entry = rho[1][2]
            phase = np.arctan2(np.imag(phase_entry), np.real(phase_entry))

        print(phase)
        ax.scatter(chi, phase, color = 'b')

    ax.set_xlabel('Chi Value')
    ax.set_ylabel('Phase')
    fig.savefig(f'{directory}/{fig_name}')
    fig.show()
