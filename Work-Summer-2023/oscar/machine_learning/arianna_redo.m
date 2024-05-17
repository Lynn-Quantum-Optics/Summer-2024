% Define symbolic variables
syms alpha theta real;

% Define states
HH = [1; 0; 0; 0];
HV = [0; 1; 0; 0];
VH = [0; 0; 1; 0];
VV = [0; 0; 0; 1];

% Define combined states
RR = (HH + 1i*HV + 1i*VH - VV) / norm(HH + 1i*HV + 1i*VH - VV);
RH = (HH + 1i*VH) / norm(HH + 1i*VH);
HR = (HH + 1i*HV) / norm(HH + 1i*HV);

% State in question
state = cos(theta)*HH + sin(theta)*exp(1i*alpha)*(RH);
state = state / norm(state);

% Matrix of the state
state_mat = state * state';
disp(state_mat);

% New W proposal
phi4_p = HH - RR;
phi4_p = phi4_p/ norm(phi4_p);

% Density matrix of the new state
phi4_mat = phi4_p * phi4_p';
disp(phi4_mat)

% partial_transpose function 
phi4_pt = partialTranspose(phi4_mat);

% Compute Wp4
Wp4 = trace(phi4_pt * state_mat);

% Extract the real part
realExpr = real(Wp4);
disp('real expr')
disp(realExpr);

initialGuess = [0,0];

% Example: Numerically explore the expression over a grid of alpha and theta values
% [AlphaGrid, ThetaGrid] = meshgrid(linspace(-pi, pi, 100), linspace(-pi/2, pi/2, 100));
% RealExprGrid = subs(realExpr, {alpha, theta}, {AlphaGrid, ThetaGrid});
% 
% disp(RealExprGrid);
% 
%  % Find instances where the expression is less than 0
% NegativeIndices = find(RealExprGrid < 0);
% 
%  if ~isempty(NegativeIndices)
%      disp('There are values of alpha and theta for which the expression is less than 0.');
% else
%      disp('No values of alpha and theta found for which the expression is less than 0.');
% end


% Convert the symbolic expression to a function handle
objFunc = matlabFunction(realExpr, 'Vars', [alpha, theta]);
% 
% Define the objective function for optimization
% Ensure this is done after defining objFunc
objectiveFunction = @(x) objFunc(x(1), x(2));
% 
% Create options for fminunc
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
% 
% Run fminunc to minimize the objective function
% Note: Directly use the adapted objectiveFunction handle
[alphaThetaOpt, objVal] = fminunc(objectiveFunction, initialGuess, options);
% 
% 
% Display the results
disp(['Optimal alpha, theta: ', num2str(alphaThetaOpt)]);
disp(['Objective function value at optimum: ', num2str(objVal)]);

