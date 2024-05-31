function x_opt = approachability(A, b)
    % APPROACHABILITY Computes the approachability solution for a matrix A and vector b.
    %   x_opt = approachability(A, b) computes the approachability solution for a given
    %   matrix A and vector b.
    %
    % Inputs:
    %   A - matrix A
    %   b - vector b
    %
    % Outputs:
    %   x_opt - approachability solution

    n = size(A, 2);
    cvx_begin quiet
        variable x(n)
        minimize( norm(A * x - b) )
    cvx_end
    x_opt = x;
end
