from rho_methods_ONLYGOOD import *
from random_gen import *
from tqdm import trange
from datetime import datetime
import sys

def measure_v_1_test(calculate=True):
    if calculate:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t.csv')
        df = df[df['concurrence'] > 0]

        # compute lynn's 
        v_1_list = []
        for i in trange(len(df)):
            row = df.iloc[i]
            # get params
            # HH, HV, HD, HA, HR, HL = all_projs[0]
            # VH, VV, VD, VA, VR, VL = all_projs[1]
            # DH, DV, DD, DA, DR, DL = all_projs[2]
            # AH, AV, AD, AA, AR, AL = all_projs[3]
            # RH, RV, RD, RA, RR, RL = all_projs[4]
            # LH, LV, LD, LA, LR, LL = all_projs[5]
            all_projs = []
            all_projs.append([row['HH'], row['HV'], row['HD'], row['HA'], row['HR'], row['HL']])    
            all_projs.append([row['VH'], row['VV'], row['VD'], row['VA'], row['VR'], row['VL']])
            all_projs.append([row['DH'], row['DV'], row['DD'], row['DA'], row['DR'], row['DL']])
            all_projs.append([row['AH'], row['AV'], row['AD'], row['AA'], row['AR'], row['AL']])
            all_projs.append([row['RH'], row['RV'], row['RD'], row['RA'], row['RR'], row['RL']])
            all_projs.append([row['LH'], row['LV'], row['LD'], row['LA'], row['LR'], row['LL']])
            rho_recons = reconstruct_rho(all_projs)
            v_1 = compute_witnesses(rho_recons, return_all=False, return_params=True)[4]
            v_1_list.append(v_1)


        # append to df
        df['v_1'] = v_1_list

        # save to csv
        df.to_csv('random_gen/data/lynn_data_400000_posconc_wlynn.csv', index=False)

    else:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t_wlynn.csv')

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    v_1 = np.where(df['v_1'] < 0, 1, 0)
    print(v_1)
    # w_witness = np.where(df['W_min'] < 0, 1, 0)
    # wp1_witness = np.where(df['Wp_t1'] < 0, 1, 0)
    # wp2_witness = np.where(df['Wp_t2'] < 0, 1, 0)
    # wp3_witness = np.where(df['Wp_t3'] < 0, 1, 0)

    print(v_1)

if __name__ == '__main__':
    measure_v_1_test()