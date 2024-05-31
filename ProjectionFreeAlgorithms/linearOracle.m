function s = linearOracle(gradient, A, b)
    % LINEARORACLE Solves the linear problem to find the update direction.
    %   s = linearOracle(gradient, A, b) solves the linear problem to find the
    %   update direction min_s <gradient, s> subject to A*s <= b.
    %
    % Inputs:
    %   gradient - gradient vector
    %   A - matrix defining the linear constraints
    %   b - vector defining the linear constraints
    %
    % Outputs:
    %   s - update direction

    K = length(b);
    f = gradient;
    options = optimoptions('linprog', 'Display', 'none');
    s = linprog(f, A, b, [], [], zeros(K, 1), [], options);
end
