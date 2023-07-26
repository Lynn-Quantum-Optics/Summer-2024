%%% Matlab version of gd_solve.m; uses HyperBell object to attempt to solve k-sys.%%%
function [coeff_ls, full_coeff_ls, nq_ls, soln_ls] = Solve(d, k, ds)
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
%      parpool(); 
    for m= 1:hb.m_limit
        fun = @(symboliv_vars)hb.get_k_sys(symboliv_vars,m);
        % Create an options object
        options = optimoptions('fsolve', 'FunctionTolerance', 10^(-hb.soln_precision), 'OptimalityTolerance', 10^(-hb.soln_precision)); % fsolve doesn't allow bound
        % need to solve and find up to 2*d orthogonal solutions
        coeff_ls = {};
        full_coeff_ls = {};
        nq_ls = {};
        soln_ls = {};
            for n = 1:1000
                [coeff, fval, exitFlag, output] = fsolve(fun, hb.get_random_guess, options);
                disp('found soln!')
%                 disp(coeff)
%                 disp(fval)
                disp(numel(coeff_ls))
                if numel(coeff_ls) == 0
                    disp('found first!')
                    coeff_ls{end+1} = coeff;
                    full_coeff_ls{end+1} = hb.get_full_coeff(coeff);
                    [n,q] = hb.convert_soln(coeff);
                    nq_ls{end+1} = [n, q];
                    soln_ls{end+1} = fval;
                else
                    if hb.is_valid_soln(coeff, coeff_ls, m)
                        disp('not valud soln')
                        coeff_ls{end+1} = coeff;
                        full_coeff_ls{end+1} = hb.get_full_coeff(coeff);
                        [n,q] = hb.convert_soln(coeff);
                        nq_ls{end+1} = [n, q];
                        soln_ls{end+1} = fval;
                    end
                end
                if numel(coeff_ls)>=hb.soln_limit
                    disp('soln lim reached!')
                    return
                end
            end
    end
% delete(gcp); % end pool session

 else % use loss function
%% get sys and solve; parallelize %%
%     parpool(); 
    for m= 1:1:hb.m_limit
        disp(m)
        coeff_ls = {}; % log real and imaginry components, what we are using to minimize
        full_coeff_ls = {}; % combine real and imaginary
        nq_ls = {}; % decompose into specific form
        soln_ls = {}; % store output of functions for reference
        for j = 1:1:1000 % try to find orthogonal solutions
            func = @(coeff) hb.loss_func(coeff, coeff_ls, m);
            coeff_guess = hb.get_random_guess();
%             [coeff, soln] = fmincon(func, coeff_guess, [], [], [], [], hb.l_bound, hb.u_bound);
            options = optimoptions('fsolve', 'FunctionTolerance', 10^(-hb.soln_precision), 'OptimalityTolerance', 10^(-hb.soln_precision)); % fsolve doesn't allow bound
            [coeff, soln, exitFlag, output] = fsolve(func, hb.get_random_guess, options);

            if hb.is_valid_soln(coeff, coeff_ls, m)
%                 disp(hb.get_full_coeff(coeff))
                coeff_ls{end+1}=coeff;
                full_coeff_ls{end+1} = hb.get_full_coeff(coeff);
                [n,q] = hb.convert_soln(coeff);
                nq_ls{end+1} = [n, q];
                soln_ls{end+1} = soln;
            end
            if numel(coeff_ls)>= hb.soln_limit
                disp('reached soln limit!')
                return;
            end
        end
    end
%     delete(gcp); % end pool session

end