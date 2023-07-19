%%% Matlab version of solve_prep.py. Defines HyperBell object to assist in the solving.%%%

classdef HyperBell
    properties
        d; % number of dimensions
        k; % number of bell states
        m; % index of particular k-group of states
        m_limit; % total number of systems to solve
        soln_limit; % number of solutions to find
        num_coeffs; % number of coefficients
        bounds; % bounds on the coefficients
        precision; % precision for solving: negative exponent. used for solution vec, orthogonality, and normalization
        k_group_indices; % indices of the k-groups of bell states
        t0; % initial time
    end
    methods
    %% constructor and setters %%
        function obj = HyperBell(d, k) % constructor
            obj.d = d;
            obj.k = k;
            obj.m = 0;
            obj.m_limit = nchoosek(d^2, k);
            obj.soln_limit = 2*d;
            obj.num_coeffs = 4*d; % splitting imag and real
            obj.bounds = {};
            for i = 1:obj.num_coeffs
                obj.bounds{end+1} = [-1, 1];
            end
            obj.precision = 6;
            obj.k_group_indices = get_k_group_indices(obj);
            obj.t0 = datetime; % log time at initilization so we can see how long it takes to solve

            disp(obj) % display the object with its params
        end
        function obj = set_precision(obj, precision)
            %%% set the precision for solving
            obj.precision = precision;
        end
        function obj = set_m(obj, m) 
            %%% choose which mth system to analyze
            obj.m = m;
        end
        function comb_unique = get_k_group_indices(obj)
            %%% construct indices of d^2 choose k states
            %% get all combinations, incl redudant (matlab doesn't have a built-in way to not include these elems)
            arr = {};
            for i = 1:obj.k
                arr{end+1} = 1:obj.d^2;
            end
            comb = arr{i};
            for i = 2:numel(arr)
                comb = combvec(comb, arr{i});
            end
            % get only unique %
            comb_unique = {};
            for i = 1:numel(comb(1, :)) % iterate through all cols
                 keep=true;
                for j = 1:numel(comb(:, i))-1 
                    
                    for l = j+1:numel(comb(:, i))
                        if comb(j, i)== comb(l, i)
                            keep=false;
                            break;
                        end
                    end 
                end
                if keep
                    % make sure other permutation of same indices doesn't
                    % already exist
                    disp(numel(comb_unique))
                    if numel(comb_unique) > 0
                        new_elem= true;
                        for o = 1:numel(comb_unique)
                            if all(unique(comb(:, i)) == unique(cell2mat(comb_unique(o))))
                                new_elem=false;
                                break;
                            end
                        end
                        if new_elem
                            comb_unique{end+1} = comb(:, i);
                            disp(cell2mat(comb(:, i)))
                        end
                    else
                        comb_unique{end+1} = comb(:, i);
                        disp(numel(comb_unique))
                    end
                end
            end
            disp(comb_unique)
        end
        %% getters for bell states, measurement, systems %%
        function bell = get_bell(obj, c, p)
            %%% get the (c,p) bell state in fock basis
            %%% c: coefficients; 0 to d-1
            %%% p: indices; 0 to d-1
            bell = zeros(2*obj.d, 1);
            for j = 1:obj.d
                phase = exp(1i*2*pi*p*(j-1)/obj.d);
                num_vec = zeros(2*obj.d, 1); % number basis portion
                num_vec(j) = 1;
                num_vec(obj.d+mod((j+c-1), obj.d)+1) = 1;
                disp(phase)
                disp(num_vec)
                bell = bell + phase*num_vec;
            end
            bell = bell * 1/sqrt(obj.d);
        end
         function v = get_full_coeff(obj, coeff)
            %%% get the full coefficient vector from the solution vector
            v = zeros(2*obj.d, 1);
            for j = 1:obj.d
                v(j) = coeff(j) + 1i*coeff(2*obj.d+j);
            end
        end
        function bell_m = get_measurement(obj, bell, coeff)
            %%% perform LELM measurement on bell state; takes in coeff vector
            bell_m = zeros(2*obj.d, 1);
            for i = 1:2*obj.d
                if bell(i) ~= 0 % if the state is occupied
                    bell_c = copy(bell);
                    bell_c(i) = 0; % annihilate the state
                    v = get_full_coeff(obj, coeff);
                    bell_m = bell_m + v*(i)*bell_c; % add to measurement
                end
            end
        end
        function ip = get_meas_ip(obj, bell1, bell2, coeff)
            %%% get the inner product of two measured states
            bell1_m = get_measurement(obj, bell1, coeff);
            bell2_m = get_measurement(obj, bell2, coeff);
            ip = bell1_m'*bell2_m;
        end
        function k_group = get_m_kgroup(obj)
            %%% get the mth k-group of bell states
            k_group_index = obj.k_group_indices(:, obj.m);
            k_group = zeros(obj.k, 1);
            for i = 1:obj.k % recover the c and p indices from the k_group_index
                c = fix(k_group_index(i)/obj.d); % integer division
                p = mod(k_group_index(i), obj.d); 
                k_group(i) = get_bell(obj, c, p);
            end
        end
        function k_sys = get_k_sys(obj, coeff)
            %%% get the k-system of bell states
            k_group = get_m_kgroup(obj);
            k_sys = zeros(numel(k_group), 1); % initialize k_sys array
            for i = 1:obj.k
                for j = i+1:obj.k
                    k_sys(i, j) = get_meas_ip(obj, k_group(i), k_group(j), coeff);
                end
            end
        end
        %% solution properties %%
        function guess = get_random_guess(obj)
            %%% get a random guess for the solution vector; normalize and thus use stick method random simplex
            r = sort(rand(obj.num_coeffs-1, 1));
            guess = zeros(obj.num_coeffs, 1);
            guess(1) = r(1); % initialize first guess element
            for i = 2:numel(r)
                guess(i) = r(i) - r(i-1); % compute "lengths" of segments
            end
            guess(numel(guess)) = 1 - r(numel(r)); % set last element
            guess = sqrt(guess);
            % randomize sign since while norm is 1, can be in range [-1, 1]
            % for phase
            for i = 1:numel(guess)
                n = rand(1) > 0.5; % if above .5, add neg sign; else keep pos
                if n
                    guess(i) = -1*guess(i);
                end
            end
        end
        function valid_soln = is_valid_soln(obj, coeff, coeff_ls)
            %%% check if the solution vector is valid.
            %%% coeff is for current found solution
            %%% coeff_ls is list of all prior found solutions
            soln_0 = all(round(obj.get_ksys(coeff), obj.precision) == 0); % check if all functions in k-system are 0
            ortho = true;
            for i = 1:numel(coeff_ls) % check if solution is orthogonal to all prior solutions
                % take inner product with all prior coeffs
                x = coeff_ls(i);
                if not(all(coeff == x)) % must not exist in list
                    ip = round(coeff'*x, obj.precision); % round inner product to precision
                    if ip ~= 0 % if not orthogonal
                        ortho = false;
                        break;
                    end
                else
                    ortho=false;
                    break;
                end
            end
            valid_soln = soln_0 & ortho; 
        end
        function [n,m] = convert_soln(obj, coeff_ls)
            %%% convert found soln to pairs of integers n and m in the form
            %%% n / d^2 e^(i 2pi m / d^2)
            % first normalize soln
            coeff_ls = coeff_ls / sum(coeff_ls.^2);
            % convert to v
            v_coeff_ls = get_full_coeff(coeff_ls);
            % get mag
            mag = abs(v_coeff_ls);
            ang = angle(v_coeff_ls);
            % convert to n and m
            n = obj.d^2*mag;
            m = ang*obj.d^2 / (2*pi);
        end
    end
end