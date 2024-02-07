% vars;
% findW;

state = 2*HH + 1i*HV + 1i*VH;
state = state / norm(state);
% disp(state);

state_mat =  state * transpose(conj(state));
% disp(state_mat)

[minW, minWp_t1, minWp_t2, minWp_t3, w_val] = findW(state_mat);
% disp(minW)
% disp(minWp_t1)
% disp(minWp_t2)
% disp(minWp_t3)

disp(w_val)