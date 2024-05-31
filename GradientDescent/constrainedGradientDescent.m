function [x, history] = constrainedGradientDescent(f, gradf, x0, alpha, maxIter, constraintFunc)
    % CONSTRAINEDGRADIENTDESCENT Performs constrained gradient descent optimization.
    %   [x, history] = constrainedGradientDescent(f, gradf, x0, alpha, maxIter, constraintFunc)
    %   optimizes the function f with its gradient gradf starting from x0 using step size alpha
    %   for maxIter iterations, subject to the constraint specified by constraintFunc.
    %
    % Inputs:
    %   f - function handle of the objective function
    %   gradf - function handle of the gradient of f
    %   x0 - initial guess
    %   alpha - step size
    %   maxIter - maximum number of iterations
    %   constraintFunc - function handle of the constraint projection function
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    
    for k = 1:maxIter
        x = x - alpha * gradf(x);
        x = constraintFunc(x);  % Project onto the constraint set
        history(k) = f(x);
    end
end
