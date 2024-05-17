from rho_methods import *
from random_gen import *
from tqdm import trange
from datetime import datetime

def get_data(num_states=1000):

    df = pd.DataFrame(columns=['concurrence', 'w', 'wp1', 'wp2', 'wp3', 'wl'])

    for _ in trange(num_states):
        rho = get_random_hurwitz()
        concurrence = get_concurrence(rho)
        w, wp1, wp2, wp3, wl = compute_witnesses(rho, return_lynn=True)
        df = pd.concat([df, pd.DataFrame({'concurrence': concurrence, 'w': w, 'wp1': wp1, 'wp2': wp2, 'wp3': wp3, 'wl': wl}, index=[0])], ignore_index=True)

    # save to csv
    time = datetime.now().strftime("%m%d%Y%H%M%S")
    df.to_csv(f'random_gen/data/lynn_data_{num_states}_{time}.csv', index=False)

    # restrict where conc > 0
    df = df[df['concurrence'] > 0]

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    # only look at where w, wp1, wp2, wp3 < 0
    w_witness = np.where(df['w'] < 0, 1, 0)
    wp1_witness = np.where(df['wp1'] < 0, 1, 0)
    wp2_witness = np.where(df['wp2'] < 0, 1, 0)
    wp3_witness = np.where(df['wp3'] < 0, 1, 0)
    lynn_witness = np.where(df['wl'] < 0, 1, 0)

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

def measure_lynn_test(calculate=True):
    if calculate:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t.csv')
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
        df['lynn_witness'] = lynn_witnesses

        # save to csv
        df.to_csv('random_gen/data/lynn_data_400000_posconc_wlynn.csv', index=False)

    else:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t_wlynn.csv')

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    lynn_witness = np.where(df['lynn_witness'] < 0, 1, 0)
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

def measure_arianna_test(calculate=True):
    '''equal magnitude HH, HV, VH, VV'''

    HH = np.array([1, 0, 0, 0]).reshape((4,1))
    HV = np.array([0, 1, 0, 0]).reshape((4,1))
    VH = np.array([0, 0, 1, 0]).reshape((4,1))
    VV = np.array([0, 0, 0, 1]).reshape((4,1))

    def get_witness(phi, rho):
            ''' Helper function to compute the witness operator for a given state and return trace(W*rho) for a given state rho.'''
            W = phi * adjoint(phi)
            W = partial_transpose(W) # take partial transpose
            return np.real(np.trace(W @ rho))

    def get_arianna_witness():
        return 1/2 * (HH + np.exp(1j*2*np.pi/4) * HV + np.exp(1j*4*np.pi/4) * VH + np.exp(1j*6*np.pi/4) * VV)

    if calculate:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t.csv')
        df = df[df['concurrence'] > 0]

        # compute lynn's 
        arianna_witnesses = []
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
            arianna_witness = get_witness(get_arianna_witness(), rho_recons)
            arianna_witnesses.append(arianna_witness)


        # append to df
        df['arianna_witness'] = arianna_witness

        # save to csv
        df.to_csv('random_gen/data/lynn_data_400000_posconc_warianna.csv', index=False)

    else:
        df = pd.read_csv('random_gen/data/roik_True_400000_r_os_t_warianna.csv')

    # check overlap of lynn's witness and w, wp1, wp2, wp3
    arianna_witness = np.where(df['arianna_witness'] < 0, 1, 0)
    w_witness = np.where(df['W_min'] < 0, 1, 0)
    wp1_witness = np.where(df['Wp_t1'] < 0, 1, 0)
    wp2_witness = np.where(df['Wp_t2'] < 0, 1, 0)
    wp3_witness = np.where(df['Wp_t3'] < 0, 1, 0)

    arianna_eff = arianna_witness.sum()/len(df)
    w_eff = w_witness.sum()/len(df)
    wp1_eff = wp1_witness.sum()/len(df)
    wp2_eff = wp2_witness.sum()/len(df)
    wp3_eff = wp3_witness.sum()/len(df)

    print('Number of states:', len(df))
    print('Arianna efficiency:', arianna_eff)
    print('W efficiency:', w_eff)
    print('Wp1 efficiency:', wp1_eff)
    print('Wp2 efficiency:', wp2_eff)
    print('Wp3 efficiency:', wp3_eff)

    # check overlap of Trues
    w_overlap = np.sum(w_witness * arianna_witness)
    wp1_overlap = np.sum(wp1_witness * arianna_witness)
    wp2_overlap = np.sum(wp2_witness * arianna_witness)
    wp3_overlap = np.sum(wp3_witness * arianna_witness)

    print('W overlap:', w_overlap/len(df))
    print('Wp1 overlap:', wp1_overlap/len(df))
    print('Wp2 overlap:', wp2_overlap/len(df))
    print('Wp3 overlap:', wp3_overlap/len(df))

if __name__ == '__main__':
    # measure_lynn_test()
    # idea = 1/2 * (HH + np.exp(1j*2*np.pi/4) * HV + np.exp(1j*4*np.pi/4) * VH + np.exp(1j*6*np.pi/4) * VV)
    measure_arianna_test()
  
