syms alpha theta real;

% Components of the equation
expr1 = 0.25 * sqrt(0.353553390593274 * exp(1.0i * alpha) * sin(2 * theta) + 1.0 + 0.353553390593274 * exp(-1.0i * alpha) * sin(2 * theta)) * exp(1.0i * alpha) * cos(theta)^2;
expr2 = 1/(sqrt(0.353553390593274 * exp(1.0i * alpha) * sin(2 * theta) + 1.0 + 0.353553390593274 * exp(-1.0i * alpha) * sin(2 * theta)));
denom = exp(1.0i * alpha) + 0.353553390593274 * exp(2.0i * alpha) * sin(2 * theta) + 0.353553390593274 * sin(2 * theta);

% Complete expression
expr = (expr1 * conj(expr2)) / denom;

% Extract the real part
realExpr = real(expr);

% Example: Numerically explore the expression over a grid of alpha and theta values
[AlphaGrid, ThetaGrid] = meshgrid(linspace(-pi, pi, 100), linspace(-pi/2, pi/2, 100));
RealExprGrid = subs(realExpr, {alpha, theta}, {AlphaGrid, ThetaGrid});

% Find instances where the expression is less than 0
NegativeIndices = find(RealExprGrid < 0);

if ~isempty(NegativeIndices)
    disp('There are values of alpha and theta for which the expression is less than 0.');
else
    disp('No values of alpha and theta found for which the expression is less than 0.');
end
