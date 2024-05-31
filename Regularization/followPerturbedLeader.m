function [x, history] = followPerturbedLeader(f, gradf, x0, alpha, sigma, maxIter)
    % FOLLOWPERTURBEDLEADER Performs Follow-The-Perturbed-Leader optimization.
    %   [x, history] = followPerturbedLeader(f, gradf, x0, alpha, sigma, maxIter) optimizes
    %   the function f with its gradient gradf starting from x0 using step size alpha,
    %   perturbation parameter sigma for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   x0 - initial guess
    %   alpha - step size
    %   sigma - perturbation parameter
    %   maxIter - maximum number of iterations
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    G = zeros(length(x0), maxIter); % Matrix to store gradients
    
    for k = 1:maxIter
        perturbation = sigma * randn(size(x0));
        G(:, k) = gradf(k, x) + perturbation;
        sumG = sum(G(:, 1:k), 2);
        x = -(alpha / k) * sumG;
        history(k) = f(k, x);
    end
end
