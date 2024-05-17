% minW: minimum witness value
% w_val: list of witness values
% set: 1=W 2=W'

function [minW, minWp_t1, minWp_t2, minWp_t3, w_val] = findW(rho)
vars;
len = 15;
T = cat(1,W1,W2,W3,W4,W5,W6, W1p,W2p,W3p,W4p,W5p,W6p,W7p,W8p,W9p);
w_val = zeros(len,1);

for i = 0:len-1
    W = T(4*i+1:4*i+4,:);
    f = matlabFunction(real(trace(W*rho)));
    if contains(char(f),'a') && contains(char(f),'b')
        x0 = [0,0,0];
        [~, fval] = fminsearch(@(x)f(x(1),x(2),x(3)), x0);
    elseif contains(char(f),'a') || contains(char(f),'b')
        x0 = [0,0];
        [~, fval] = fminsearch(@(x)f(x(1),x(2)), x0);
    elseif contains(char(f),'t')
        [~,fval] = fminbnd(f,0,pi);
    else
        fval = trace(W*rho);
    end
    
    if abs(fval) < 1e-5
        fval = 0;
    end

    w_val(i+1,1) = fval;
end
minW = min(w_val(1:6));
minWp_t1 = min(w_val(7:9));
minWp_t2 = min(w_val(10:12));
minWp_t3 = min(w_val(13:15));
end