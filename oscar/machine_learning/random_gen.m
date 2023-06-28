function [rho, params] = random_gen(log_params)
r = rand(1,3);
M11 = r(1,1);
M22 = r(1,2)*(1-M11);
M33 = r(1,3)*(1-M11-M22);
M44 = 1-M11-M22-M33;

dia = [M11 M22 M33 M44];
dia = dia(1,randperm(4));

M = diag(dia);

U = zeros(0);
params = zeros(0);
for j = 1:6
    t = asin(sqrt(unifrnd(0,1)));
    a = unifrnd(0,2*pi);
    x = unifrnd(0,2*pi);
    y = unifrnd(0,2*pi);
    Uj = exp(1i*a)*[exp(1i*x)*cos(t) exp(1i*y)*sin(t); -exp(-1i*y)*sin(t) exp(-1i*x)*cos(t)];
    U = cat(1,U,Uj);
    if log_params ==true && j==4
        params=cat(1, params, a, x, y, t);
    end
end

U1 = U(1:2,:);
K1 = [1 0 0 0; 0 1 0 0; 0 0 U1(1,1) U1(1,2); 0 0 U1(2,1) U1(2,2)];
U2 = U(3:4,:);
K2 = [1 0 0 0; 0 U2(1,1) U2(1,2) 0; 0 U2(2,1) U2(2,2) 0; 0 0 0 1];
U3 = U(5:6,:);
K3 = [U3(1,1) U3(1,2) 0 0; U3(2,1) U3(2,2) 0 0; 0 0 1 0; 0 0 0 1];
U4 = U(7:8,:);
K4 = [1 0 0 0; 0 1 0 0; 0 0 U4(1,1) U4(1,2); 0 0 U4(2,1) U4(2,2)];
U5 = U(9:10,:);
K5 = [1 0 0 0; 0 U5(1,1) U5(1,2) 0; 0 U5(2,1) U5(2,2) 0; 0 0 0 1];
U6 = U(11:12,:);
K6 = [1 0 0 0; 0 1 0 0; 0 0 U6(1,1) U6(1,2); 0 0 U6(2,1) U6(2,2)];

K = K1*K2*K3*K4*K5*K6;

rho = K*M*K';

end
