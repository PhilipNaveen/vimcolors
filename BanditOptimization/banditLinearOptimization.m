function [x, history] = banditLinearOptimization(f, A, b, x0, alpha, maxIter, delta)
    % BANDITLINEAROPTIMIZATION Performs bandit optimization for linear functions.
    %   [x, history] = banditLinearOptimization(f, A, b, x0, alpha, maxIter, delta) optimizes
    %   the function f defined by A and b starting from x0 using step size alpha for maxIter iterations
    %   with perturbation parameter delta.
    %
    % Inputs:
    %   f - function handle of the objective function
    %   A - matrix defining the linear function
    %   b - vector defining the linear function
    %   x0 - initial guess
    %   alpha - step size
    %   maxIter - maximum number of iterations
    %   delta - perturbation parameter
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    n = length(x0);
    
    for k = 1:maxIter
        u = randn(n, 1);
        u = u / norm(u);  % Unit vector
        fk_plus = f(x + delta * u, A, b);
        fk_minus = f(x - delta * u, A, b);
        gradEstimate = (fk_plus - fk_minus) / (2 * delta) * u;
        x = x - alpha * gradEstimate;
        history(k) = f(x, A, b);
    end
end

function value = linearFunction(x, A, b)
    % Linear function value calculation
    value = A * x + b;
end
