%%% Matlab version of gd_solve.m; uses HyperBell object to attempt to solve k-sys.%%%
function solns = Solve(d, k, ds)
%%% options: 
%%%         d = int, dimension
%%%         k = int, subset of d^2 bell states
%%%         ds: bool, whether to try direct solution; if false, tries to
%%%         minimize loss function
hb = HyperBell(d,k);
if ds: % attempt direct numerical solve
    %% get vars%%
    % loop to create string names
    var_names = {};
    for i = 1:2*d
        var_names{end+1} = sprintf('v%d', i);  % Generate variable name
    end
    % loop to create symbolic vars
    symbolic_vars = cell(1, numel(var_names));
    for i = 1:numel(symbolic_vars)
        symbolic_vars{i} = sym(var_names{i});
    end
    % convert to comma separated list
    var_ls = cell2mat(symbolic_vars);
    % pass as input to syms
    sys(var_ls);

    %% get sys and solve; parallelize %%
    parpool(); 
    parfor i= 1:

end
end