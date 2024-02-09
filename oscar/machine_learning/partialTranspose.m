function rhoPT = partialTranspose(rho)
    % This function computes the partial transpose of a 4x4 density matrix
    % on the second subsystem.
    %
    % Parameters:
    % rho - a 4x4 density matrix of a bipartite quantum system
    %
    % Returns:
    % rhoPT - partial transpose of rho with respect to the second subsystem

    % Initialize the partial transpose matrix
    rhoPT = zeros(4, 4);
    
    % Mapping for partial transpose on the second subsystem in a 4x4 matrix
    % Essentially, we're swapping the indices (2,3) and (3,2) while
    % leaving the rest unchanged.
    map = [1, 3, 2, 4;  % Mapping for rows
           1, 3, 2, 4]; % Mapping for columns

    % Perform the partial transpose operation
    for i = 1:4
        for j = 1:4
            rhoPT(i, j) = rho(map(1, i), map(2, j));
        end
    end
end
