%% load the states %%
data = table();
n = 10;
parpool(); 
parfor i= 1:n
% for i = 1:n
    disp(i)
    %% get rho and params %%
    [rho, params] = Random(true); % true/false to log generating unitary parameters
    alpha = params(1);
    chi = params(2);
    psi = params(3);
    phi = params(4);
    %% concurrence, purity, min eig of PT %%
    concurrence = Concurrence(rho);
    purity = Purity(rho);
    min_eig= MinEig(rho);
     [W_min, Wp_t1, Wp_t2, Wp_t3] = findW(rho);
     %% projections %%
     probs = Projection(rho);
     HH = probs(1,1);
    HV = probs(1,2);
    HD = probs(1,3);
    HA = probs(1,4);
    HR = probs(1,5);
    HL = probs(1,6);
    VH = probs(2,1);
    VV = probs(2,2);
    VD = probs(2,3);
    VA = probs(2,4);
    VR = probs(2,5);
    VL = probs(2,6);
    DH = probs(3,1);
    DV = probs(3,2);
    DD = probs(3,3);
    DA = probs(3,4);
    DR = probs(3,5);
    DL = probs(3,6);
    AH = probs(4,1);
    AV = probs(4,2);
    AD = probs(4,3);
    AA = probs(4,4);
    AR = probs(4,5);
    AL = probs(4,6);
    RH = probs(5,1);
    RV = probs(5,2);
    RD = probs(5,3);
    RA = probs(5,4);
    RR = probs(5,5);
    RL = probs(5,6);
    LH = probs(6,1);
    LV = probs(6,2);
    LD = probs(6,3);
    LA = probs(6,4);
    LR = probs(6,5);
    LL = probs(6,6);
%     disp('------')  
%     disp(DD + DA + AD + AA);
%     disp('------')
%% append to df %%
     row = table(HH, HV, HD, HA, HR, HL, VH, VV, VD, VA, VR, VL, DH, DV, DD, DA, DR, DL, AH, AV, AD, AA, AR, AL, RH, RV, RD, RA, RR, RL, LH, LV, LD, LA, LR, LL, concurrence, purity, W_min, Wp_t1, Wp_t2, Wp_t3, min_eig, alpha, chi, psi, phi);
    data = [data; row];
end
% save states for processing
cd('/Users/oscarscholin/Desktop/Pomona/Summer23/Summer-2023/oscar/machine_learning/random_gen/test')
writetable(data, sprintf('%d_tt_m.csv', n));
delete(gcp);