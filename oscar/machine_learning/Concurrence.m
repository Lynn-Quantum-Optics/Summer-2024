function concurrence = Concurrence(rho)
sy = [0 -1i;1i 0];
sig = kron(sy, sy);
R = sqrtm(sqrtm(rho) * sig * conj(rho) * sig * sqrtm(rho)); % correct defintion
    
evals = eig(R);
evals = sort(evals, 'descend');
concurrence = max(0, real(evals(1)-evals(2)-evals(3)-evals(4)));

end
