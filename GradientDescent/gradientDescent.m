function [x, history] = gradientDescent(f, gradf, x0, alpha, maxIter)
    % GRADIENTDESCENT Performs gradient descent optimization.
    %   [x, history] = gradientDescent(f, gradf, x0, alpha, maxIter) optimizes
    %   the function f with its gradient gradf starting from x0 using step size alpha
    %   for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function
    %   gradf - function handle of the gradient of f
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
        x = x - alpha * gradf(x);
        history(k) = f(x);
    end
end
