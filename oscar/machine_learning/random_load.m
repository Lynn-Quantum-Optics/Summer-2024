%% load the states %%
data = table();
n = 100000
parpool(); 
parfor i= 1:n
    disp(i)
    [rho, params] = random_gen(true);
    alpha = params(1);
    chi = params(2);
    psi = params(3);
    phi = params(4);
    concurrence = Concurrence(rho);
    purity = Purity(rho);
    min_eig= MinEig(rho);
     [W_min, Wp_t1, Wp_t2, Wp_t3] = findW(rho);
     row = table(concurrence, purity, W_min, Wp_t1, Wp_t2, Wp_t3, alpha, chi, psi, phi);
%     row = table(concurrence, purity);
    data = [data; row];
end
% save states for processing
cd('/Users/oscarscholin/Desktop/Pomona/Summer23/Summer-2023/oscar/machine_learning/random_gen/test')
writetable(data, sprintf('%d_m.csv', n));
delete(gcp);