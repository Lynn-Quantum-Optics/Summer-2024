%%% Matlab version of solve_prep.py. Defines HyperBell object to assist in the solving.%%%

classdef HyperBell
    properties
        d; % number of dimensions
        k; % number of bell states
        m_limit; % total number of systems to solve
        soln_limit; % number of solutions to find
        num_coeffs; % number of coefficients
        l_bound; % lower bound
        u_bound; % upper bound
        bounds; % bounds on the coefficients
        soln_precision; % precision for solving: negative exponent. used for solution vec, orthogonality, and normalization
        coeff_precision;
        t0; % initial time
    end
    methods
    %% constructor and setters %%
        function obj = HyperBell(d, k) % constructor
            obj.d = d;
            obj.k = k;
            obj.m_limit = nchoosek(d^2, k);
            obj.soln_limit = 2*d;
            obj.num_coeffs = 4*d; % splitting imag and real
            obj.l_bound = zeros(obj.num_coeffs, 1);
            for i = 1:obj.num_coeffs
                obj.l_bound(i) = -1;
            end
            obj.u_bound = zeros(obj.num_coeffs, 1);
            for i = 1:obj.num_coeffs
                obj.u_bound(i) = 1;
            end
            obj.bounds = {};
            for i = 1:obj.num_coeffs
                obj.bounds{end+1} = [-1; 1];
            end
            obj.soln_precision = 6;
            obj.coeff_precision = 3;
            obj.t0 = datetime; % log time at initilization so we can see how long it takes to solve

            disp(obj) % display the object with its params
        end
        function obj = set_precision(obj, precision)
            %%% set the precision for solving
            obj.precision = precision;
        end
        function m_k_sys = get_m_k_sys(obj, m)
            %%% gets the mth combination of d^2 choose k elements; uses CNS
            %%% (combinatorial number system)
            % m starts at 1, but need to reset by -1 for alg
            if m > nchoosek(obj.d^2, obj.k)
                error fprintf("m=%d exceeds d^2=%d", m, obj.d^2);
            end
            m = m-1;
            function [largest_val, ind] = do_round(targ, n)
                %%% finds the largest value of d^2 choose n to targ
                if targ==0
                    largest_val=0;
                    ind=n-1;
                    return
                elseif targ==1
                    largest_val = 0;
                    ind = n;
                end
                i = -1;
                largest_val=0;
                while (largest_val <= targ) & (n+1+i <= obj.d^2)
                    largest_val = nchoosek(n+1+i, n);
                    if largest_val > targ
                        i= i-1;
                        ind = n+i+1; % get the index corresponding to largest_val
                        if n+1+i == n-1
                            largest_val= 0;
                            ind = n-1;
                        else
                            largest_val = nchoosek(n+1+i, n);
                            ind = n+1+i;
                        end
                        return
                    end
                    i = i+1;
                end
                ind = n+i+1;
            end
            targ= m; % largest val starts at m
            m_k_sys = zeros(obj.k,1);
            for j = obj.k:-1:1
                [largest_val, ind] = do_round(targ, j);
                m_k_sys(j) = ind;
                targ=abs(targ-largest_val);
            end
            m_k_sys = m_k_sys +1; % offset by +1 so alligns with matlab indicies
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
                if bell(i)~= 0 % if the state is occupied
                    bell_c = bell;
                    bell_c(i) = 0; % annihilate the state
                    v = get_full_coeff(obj, coeff);
                    bell_m = bell_m + v(i)*bell_c; % add to measurement
                end
            end
        end
        function ip = get_meas_ip(obj, bell1, bell2, coeff)
            %%% get the inner product of two measured states
            bell1_m = get_measurement(obj, cell2mat(bell1), coeff);
            bell2_m = get_measurement(obj, cell2mat(bell2), coeff);
            ip = bell1_m'*bell2_m;
        end
        function k_group = get_m_kgroup(obj, m)
            %%% get the mth k-group of bell states
            k_group_indices = obj.get_m_k_sys(m);
            k_group = {};
            for i = 1:numel(k_group_indices) % recover the c and p indices from the k_group_index
                ind= k_group_indices(i);
                c = fix(ind/obj.d); % integer division
                p = mod(ind, obj.d); 
                k_group{i} = get_bell(obj, c, p);
            end
        end
        function k_sys = get_k_sys(obj, coeff, m)
            %%% get the k-system of bell states
            k_group = get_m_kgroup(obj, m);
            k_sys = zeros(numel(k_group), 1); % initialize k_sys array
            for i = 1:obj.k
                for j = i+1:obj.k
                    k_sys(i) = get_meas_ip(obj, k_group(i), k_group(j), coeff);
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
        function coeff_ip_ls = get_coeff_ip_ls(obj, coeff, coeff_ls)
            %%% returns list of inner products of coeff with coeff_ls
                coeff_ip_ls = zeros(numel(coeff_ls),1);
                coeff = obj.get_full_coeff(coeff);
                for i = 1:numel(coeff_ls) % check if solution is orthogonal to all prior solutions
                    % take inner product with all prior coeffs
                    x = obj.get_full_coeff(cell2mat(coeff_ls(i)));
                    if not(all(coeff == x)) % must not exist in list
                        ip = coeff'*x;
                        coeff_ip_ls(i) = ip;
                    else
                        coeff_ip_ls(i)=100; % punish model if chooses duplicate state
                    end
                end
        end
        function ortho = is_ortho_coeff(obj, coeff, coeff_ls)
            %%% uses output from get_coeff_ip_ls
            ortho=true;
            coeff_ip_ls = obj.get_coeff_ip_ls(coeff, coeff_ls);
            if any(abs(round(coeff_ip_ls, obj.soln_precision))>0) % check if any ip have magntude too great
                ortho=false;
            end
        end
        function valid_soln = is_valid_soln(obj, coeff, coeff_ls, m)
            %%% check if the solution vector is valid.
            %%% coeff is for current found solution
            %%% coeff_ls is list of all prior found solutions
            coeff_0 = not(all(round(obj.get_full_coeff(coeff), obj.coeff_precision) == 0));
            soln_0 = all(round(obj.get_k_sys(coeff, m), obj.soln_precision) == 0); % check if all functions in k-system are 0
            if numel(coeff_ls)>1
                ortho = obj.is_ortho_coeff(coeff, coeff_ls);
                valid_soln = coeff_0 & soln_0 & ortho; 
            else
                 valid_soln = coeff_0 & soln_0; % don't need to check orthogonal if this is first solution
            end
        end
        function [n,m] = convert_soln(obj, coeff)
            %%% convert found soln to pairs of integers n and m in the form
            %%% n / d^2 e^(i 2pi m / d^2)
            % first normalize soln
            coeff = coeff / sum(coeff.^2);
            % convert to v
            v_coeff_ls = obj.get_full_coeff(coeff);
            % get mag
            mag = abs(v_coeff_ls);
            ang = angle(v_coeff_ls);
            % convert to n and m
            n = obj.d^2*mag;
            m = ang*obj.d^2 / (2*pi);
        end
        function loss = loss_func(obj, coeff, coeff_ls, m)
         %%% three parts: 
         % 1. absolute difference in norm from 1
         % 2. RSE for function values
         % 3. RSE for inner products with exisiting solutions
%          coeff = cell2mat(coeff);
%          coeff_ls = cell2mat(coeff_ls);
         coeff_v = get_full_coeff(obj, coeff);
         l1 = sqrt(abs(sum(coeff_v.^2)));
         l2 = sqrt(abs(sum(get_k_sys(obj, coeff, m).^2)));
         l3 = sqrt(abs(sum(get_coeff_ip_ls(obj, coeff, coeff_ls).^2)));
         % sum all 3 together to get complete loss
         loss = l1+l2+l3;
        end
    end
end