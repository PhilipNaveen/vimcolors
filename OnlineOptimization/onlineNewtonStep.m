function [x, history] = onlineNewtonStep(f, gradf, hessf, x0, alpha, maxIter)
    % ONLINENEWTONSTEP Performs online Newton step optimization.
    %   [x, history] = onlineNewtonStep(f, gradf, hessf, x0, alpha, maxIter) optimizes
    %   the function f with its gradient gradf and Hessian hessf starting from x0 using step size alpha
    %   for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   hessf - function handle of the Hessian of f (sequence of Hessians)
    %   x0 - initial guess
    %   alpha - step size
    %   maxIter - maximum number of iterations
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    
    for k = 1:maxIter
        fk = f(k, x);  % Current function
        gradfk = gradf(k, x);  % Current gradient
        hessfk = hessf(k, x);  % Current Hessian
        x = x - alpha * (hessfk \ gradfk);
        history(k) = fk;
    end
end
