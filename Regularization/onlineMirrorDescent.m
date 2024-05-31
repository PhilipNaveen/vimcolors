function [x, history] = onlineMirrorDescent(f, gradf, x0, alpha, maxIter, mirrorMap, invMirrorMap)
    % ONLINEMIRRORDESCENT Performs online mirror descent optimization.
    %   [x, history] = onlineMirrorDescent(f, gradf, x0, alpha, maxIter, mirrorMap, invMirrorMap) optimizes
    %   the function f with its gradient gradf starting from x0 using step size alpha
    %   for maxIter iterations with given mirror map and its inverse.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   x0 - initial guess
    %   alpha - step size
    %   maxIter - maximum number of iterations
    %   mirrorMap - mirror map function handle
    %   invMirrorMap - inverse mirror map function handle
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    
    for k = 1:maxIter
        fk = f(k, x);  % Current function
        gradfk = gradf(k, x);  % Current gradient
        z = mirrorMap(x) - alpha * gradfk;
        x = invMirrorMap(z);
        history(k) = fk;
    end
end
