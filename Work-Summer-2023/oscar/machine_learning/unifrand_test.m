unifrand_ls = [];
for i = 1:100000
    unifrand_ls = [unifrand_ls, unifrnd(0,2*pi)];
end
cd('/Users/oscarscholin/Desktop/Pomona/Summer23/Summer-2023/oscar/machine_learning/random_gen/test')
save('unifrand_ls.mat', 'unifrand_ls');