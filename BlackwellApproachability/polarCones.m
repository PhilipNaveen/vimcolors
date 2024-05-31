function [cones, dualCones] = polarCones(A, b)
    % POLARCONES Computes the polar cones for a matrix A and vector b.
    %   [cones, dualCones] = polarCones(A, b) computes the polar cones for a given
    %   matrix A and vector b.
    %
    % Inputs:
    %   A - matrix A
    %   b - vector b
    %
    % Outputs:
    %   cones - polar cones
    %   dualCones - dual polar cones

    [m, n] = size(A);
    cones = cell(n, 1);
    dualCones = cell(m, 1);

    for i = 1:n
        % Define cone for each column of A
        cvx_begin quiet
            variable x(n)
            minimize( norm(A * x - b) )
            subject to
                norm(x(i+1:end)) <= x(i);
        cvx_end
        cones{i} = x;
    end

    for j = 1:m
        % Define dual cone for each row of A
        cvx_begin quiet
            variable y(m)
            maximize( y' * b )
            subject to
                y' * A <= 0;
                y(j) == 1;
        cvx_end
        dualCones{j} = y;
    end
end
