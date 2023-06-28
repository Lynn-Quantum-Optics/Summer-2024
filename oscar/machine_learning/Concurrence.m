function concurrence = Concurrence(rho)
vars;
R = sqrtm(sqrtm(rho) * sig * conj(rho) * sig * sqrtm(rho));
evals = eig(R);
evals = sort(evals, 'descend');
% disp(evals);
concurrence = real(max(0, evals(1)-evals(2)-evals(3)-evals(4)));
end
