function [x, history] = adaptiveGradientDescent(f, gradf, x0, alpha, beta, maxIter)
    % ADAPTIVEGRADIENTDESCENT Performs adaptive gradient descent optimization.
    %   [x, history] = adaptiveGradientDescent(f, gradf, x0, alpha, beta, maxIter) optimizes
    %   the function f with its gradient gradf starting from x0 using initial step size alpha
    %   and adaptation parameter beta for maxIter iterations.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   x0 - initial guess
    %   alpha - initial step size
    %   beta - adaptation parameter
    %   maxIter - maximum number of iterations
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    gradHist = zeros(size(x0));
    
    for k = 1:maxIter
        fk = f(k, x);  % Current function
        gradfk = gradf(k, x);  % Current gradient
        gradHist = gradHist + gradfk.^2;
        adjAlpha = alpha ./ (sqrt(gradHist) + beta);
        x = x - adjAlpha .* gradfk;
        history(k) = fk;
    end
end
