from rho_methods_ONLYGOOD_oscar import *
from random_gen import *
from tqdm import trange
from datetime import datetime

def get_data(num_states=1000):

    df = pd.DataFrame(columns=['concurrence', 'w', 'wp1', 'wp2', 'wp3', 'wv1'])

    for _ in trange(num_states):
        rho = get_random_hurwitz()
        concurrence = get_concurrence(rho)
        w, wp1, wp2, wp3, wv1 = compute_witnesses(rho, compare_arianna=True)
        df = pd.concat([df, pd.DataFrame({'concurrence': concurrence, 'w': w, 'wp1': wp1, 'wp2': wp2, 'wp3': wp3, 'wv1': wv1}, index=[0])], ignore_index=True)

    # save to csv
    time = datetime.now().strftime("%m%d%Y%H%M%S")
    df.to_csv(f'arianna_data_{num_states}_{time}.csv', index=False)

    # restrict where conc > 0
    df = df[df['concurrence'] > 0]

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    # only look at where w, wp1, wp2, wp3 < 0
    w_witness = np.where(df['w'] < 0, 1, 0)
    wp1_witness = np.where(df['wp1'] < 0, 1, 0)
    wp2_witness = np.where(df['wp2'] < 0, 1, 0)
    wp3_witness = np.where(df['wp3'] < 0, 1, 0)
    v1_witness = np.where(df['wv1'] < 0, 1, 0)

    v1_eff = v1_witness.sum()/len(df)
    w_eff = w_witness.sum()/len(df)
    wp1_eff = wp1_witness.sum()/len(df)
    wp2_eff = wp2_witness.sum()/len(df)
    wp3_eff = wp3_witness.sum()/len(df)

    print('Number of states:', len(df))
    print('v1 efficiency:', v1_eff)
    print('W efficiency:', w_eff)
    print('Wp1 efficiency:', wp1_eff)
    print('Wp2 efficiency:', wp2_eff)
    print('Wp3 efficiency:', wp3_eff)

    # check overlap of Trues
    w_overlap = np.sum(w_witness * v1_witness)
    wp1_overlap = np.sum(wp1_witness * v1_witness)
    wp2_overlap = np.sum(wp2_witness * v1_witness)
    wp3_overlap = np.sum(wp3_witness * v1_witness)

    print('W overlap:', w_overlap/len(df))
    print('Wp1 overlap:', wp1_overlap/len(df))
    print('Wp2 overlap:', wp2_overlap/len(df))
    print('Wp3 overlap:', wp3_overlap/len(df))

def measure_arianna_test(calculate=True):
    if calculate:
        df = pd.read_csv('roik_True_400000_r_os_t.csv')
        df = df[df['concurrence'] > 0]

        # compute lynn's 
        lynn_witnesses = []
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
            lynn_witness = compute_witnesses(rho_recons, return_lynn_only=True)
            lynn_witnesses.append(lynn_witness)

        # append to df
        df['arianna_witness'] = lynn_witnesses

        # save to csv
        df.to_csv('lynn_data_400000_posconc_warianna.csv', index=False)

    else:
        df = pd.read_csv('lynn_data_400000_posconc_warianna.csv')

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    lynn_witness = np.where(df['arianna_witness'] < 0, 1, 0)
    w_witness = np.where(df['W_min'] < 0, 1, 0)
    wp1_witness = np.where(df['Wp_t1'] < 0, 1, 0)
    wp2_witness = np.where(df['Wp_t2'] < 0, 1, 0)
    wp3_witness = np.where(df['Wp_t3'] < 0, 1, 0)

    lynn_eff = lynn_witness.sum()/len(df)
    w_eff = w_witness.sum()/len(df)
    wp1_eff = wp1_witness.sum()/len(df)
    wp2_eff = wp2_witness.sum()/len(df)
    wp3_eff = wp3_witness.sum()/len(df)

    print('Number of states:', len(df))
    print('Lynn efficiency:', lynn_eff)
    print('W efficiency:', w_eff)
    print('Wp1 efficiency:', wp1_eff)
    print('Wp2 efficiency:', wp2_eff)
    print('Wp3 efficiency:', wp3_eff)

    # check overlap of Trues
    w_overlap = np.sum(w_witness * lynn_witness)
    wp1_overlap = np.sum(wp1_witness * lynn_witness)
    wp2_overlap = np.sum(wp2_witness * lynn_witness)
    wp3_overlap = np.sum(wp3_witness * lynn_witness)

    print('W overlap:', w_overlap/len(df))
    print('Wp1 overlap:', wp1_overlap/len(df))
    print('Wp2 overlap:', wp2_overlap/len(df))
    print('Wp3 overlap:', wp3_overlap/len(df))

if __name__ == '__main__':
    measure_arianna_test()
  

