phiP = 1/sqrt(2)*[1;0;0;1];
phiM = 1/sqrt(2)*[1;0;0;-1];
psiP = 1/sqrt(2)*[0;1;1;0];
psiM = 1/sqrt(2)*[0;1;-1;0];

% Test cases
% t1 = 1/sqrt(2)*(psiP + exp(1i*pi/3)*psiM);
% t2 = 1/sqrt(2)*(phiP + exp(8i*pi/9)*phiM);
% t3 = 1/sqrt(2)*(phiM + exp(1i*pi/2)*psiM);
% t4 = 1/sqrt(2)*(phiP + exp(1i*pi/2)*psiP);
% t5 = 1/sqrt(2)*(phiP + exp(0*1i*pi/2)*psiM);
% t6 = 1/sqrt(2)*(phiM + exp(1i*pi/2)*psiP);

% Pauli Tensor
sig0 = [1 0;0 1];
sig1 = [0 1;1 0];
sig2 = [0 -1i;1i 0];
sig3 = [1 0;0 -1];

ii = kron(sig0,sig0);
ix = kron(sig0,sig1);
iy = kron(sig0,sig2);
iz = kron(sig0,sig3);
xi = kron(sig1,sig0);
xx = kron(sig1,sig1);
xy = kron(sig1,sig2);
xz = kron(sig1,sig3);
yi = kron(sig2,sig0);
yx = kron(sig2,sig1);
yy = kron(sig2,sig2);
yz = kron(sig2,sig3);
zi = kron(sig3,sig0);
zx = kron(sig3,sig1);
zy = kron(sig3,sig2);
zz = kron(sig3,sig3);

syms a b t real
sig = [0 0 0 -1;0 0 1 0;0 1 0 0;-1 0 0 0];

% --------------------- W ---------------------

P1 = cos(t)*phiP + sin(t)*phiM; 
W1 = partialTrans(P1*P1');

P2 = cos(t)*psiP + sin(t)*psiM;
W2 = partialTrans(P2*P2');

P3 = cos(t)*phiP + sin(t)*psiP;
W3 = partialTrans(P3*P3');

P4 = cos(t)*phiM + sin(t)*psiM;
W4 = partialTrans(P4*P4');

P5 = cos(t)*phiP + 1i*sin(t)*psiM;
W5 = partialTrans(P5*P5');

P6 = cos(t)*phiM + 1i*sin(t)*psiP;
W6 = partialTrans(P6*P6');

% --------------------- W' ---------------------

% Pair 1
P1p = 1/sqrt(2)*[cos(t)+exp(1i*a)*sin(t);0;0;cos(t)-exp(1i*a)*sin(t)];
W1p = partialTrans(P1p*P1p');

P2p = 1/sqrt(2)*[0;cos(t)+exp(1i*a)*sin(t);cos(t)-exp(1i*a)*sin(t);0];
W2p = partialTrans(P2p*P2p');

P3p = 1/sqrt(2)*[cos(t);sin(t)*exp(1i*(b-a));sin(t)*exp(1i*a);cos(t)*exp(1i*b)];
W3p = partialTrans(P3p*P3p');

% Pair 2
P4p = 1/sqrt(2)*[cos(t);sin(t)*exp(1i*a);sin(t)*exp(1i*a);cos(t)];
W4p = partialTrans(P4p*P4p');

P5p = 1/sqrt(2)*[cos(t);sin(t)*exp(1i*a);-sin(t)*exp(1i*a);-cos(t)];
W5p = partialTrans(P5p*P5p');

P6p = [cos(t)*cos(a);1i*cos(t)*sin(a);1i*sin(t)*sin(b);sin(t)*cos(b)];
W6p = partialTrans(P6p*P6p');

% Pair 3
P7p = 1/sqrt(2)*[cos(t);sin(t)*exp(1i*a);-sin(t)*exp(1i*a);cos(t)];
W7p = partialTrans(P7p*P7p');

P8p = 1/sqrt(2)*[cos(t);sin(t)*exp(1i*a);sin(t)*exp(1i*a);-cos(t)];
W8p = partialTrans(P8p*P8p');

P9p = [cos(t)*cos(a);cos(t)*sin(a);sin(t)*sin(b);sin(t)*cos(b)];
W9p = partialTrans(P9p*P9p');