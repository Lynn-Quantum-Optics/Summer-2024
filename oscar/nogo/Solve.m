%%% Matlab version of gd_solve.m; uses HyperBell object to attempt to solve k-sys.%%%
function solns = Solve(d, k, ds)
%%% options: 
%%%         d = int, dimension
%%%         k = int, subset of d^2 bell states
%%%         ds: bool, whether to try direct solution; if false, tries to
%%%         minimize loss function
hb = HyperBell(d,k);
if ds % attempt direct numerical solve
    %% get vars%%
    % loop to create string names
    var_names = {};
     for i = 1:hb.num_coeffs
         var_names{end+1} = sprintf('x%d', i);  % Generate variable name
     end
     % loop to create symbolic vars
    symbolic_vars = sym('x',[1, numel(var_names)]);
    % pass as input to syms
     for i = 1:numel(var_names)
        symbolic_vars(i) = sym(var_names{i});
     end

    %% get sys and solve; parallelize %%
    parpool(); 
    parfor m= 1:hb.m_limit
        fun = @(symboliv_vars)hb.get_k_sys(symboliv_vars,m);
        % Create an options object
        options = optimoptions('fsolve', 'FunctionTolerance', 1e-9); % fsolve doesn't allow bound
        % need to solve and find up to 2*d orthogonal solutions
        S= fsolve(fun, hb.get_random_guess, options);
        disp(S);
    end
    delete(gcp); % end pool session

end
end