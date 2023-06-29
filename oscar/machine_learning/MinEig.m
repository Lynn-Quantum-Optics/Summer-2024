function min_eig = MinEig(rho)
PT = partialTrans(rho);
eig_vals = eig(PT);
eig_vals = sort(eig_vals, 'ascend');
min_eig = real(eig_vals(1));
end