function [x, history] = rftlAlgorithm(f, gradf, x0, alpha, lambda, maxIter)
    % RFTLALGORITHM Performs the Regularized Follow-The-Leader (RFTL) optimization.
    %   [x, history] = rftlAlgorithm(f, gradf, x0, alpha, lambda, maxIter) optimizes
    %   the function f with its gradient gradf starting from x0 using step size alpha
    %   and regularization parameter lambda for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   x0 - initial guess
    %   alpha - step size
    %   lambda - regularization parameter
    %   maxIter - maximum number of iterations
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    G = zeros(length(x0), maxIter); % Matrix to store gradients
    
    for k = 1:maxIter
        G(:, k) = gradf(k, x);
        sumG = sum(G(:, 1:k), 2);
        x = -(alpha / (k + lambda)) * sumG;
        history(k) = f(k, x);
    end
end
