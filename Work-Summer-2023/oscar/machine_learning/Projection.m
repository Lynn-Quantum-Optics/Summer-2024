function probs = Projection(rho)
%% definitions of basis states %%
H = [1;0];
V = [0; 1];
D = [1/sqrt(2); 1/sqrt(2)];
A = [1/sqrt(2); -1/sqrt(2)];
R = [1/sqrt(2); 1j/sqrt(2)];
L = [1/sqrt(2); -1j/sqrt(2)];
%% projection operator%%
    function prob = compute_proj(b1, b2, rho)
        proj1 = b1 * ctranspose(b1);
        proj2 = b2 * ctranspose(b2);
        prob = real(trace(kron(proj1, proj2) * rho));
    end
%% compute all projections %%
basis_ls = {H, V, D, A, R, L};
probs = [];
for l=1:numel(basis_ls)
    b1 = basis_ls{l};
    for m=1:numel(basis_ls)
        b2 = basis_ls{m};
        % compute proj
        prob = compute_proj(b1, b2, rho);
        probs = horzcat(probs, prob);
    end
end
%% resize into array %%
probs = reshape(probs, 6, 6);
end