function [x, history] = onlineGradientDescent(f, gradf, x0, alpha, maxIter)
    % ONLINEGRADIENTDESCENT Performs online gradient descent optimization.
    %   [x, history] = onlineGradientDescent(f, gradf, x0, alpha, maxIter) optimizes
    %   the function f with its gradient gradf starting from x0 using step size alpha
    %   for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
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
        x = x - alpha * gradfk;
        history(k) = fk;
    end
end
