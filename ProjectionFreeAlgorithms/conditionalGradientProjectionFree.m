function [x, history] = conditionalGradientProjectionFree(f, gradf, x0, A, b, maxIter, tol)
    % CONDITIONALGRADIENTPROJECTIONFREE Performs the projection-free conditional gradient method.
    %   [x, history] = conditionalGradientProjectionFree(f, gradf, x0, A, b, maxIter, tol) optimizes
    %   the function f with gradient gradf starting from x0 using the linear constraints
    %   defined by A and b for maxIter iterations with tolerance tol.
    %
    % Inputs:
    %   f - function handle of the objective function
    %   gradf - function handle of the gradient of the objective function
    %   x0 - initial guess
    %   A - matrix defining the linear constraints
    %   b - vector defining the linear constraints
    %   maxIter - maximum number of iterations
    %   tol - tolerance for convergence
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    
    for k = 1:maxIter
        gradient = gradf(x);
        % Solve the linear problem: s = argmin_s <gradf(x), s> s.t. A*s <= b
        s = linearOracle(gradient, A, b);
        % Line search
        alpha = 2 / (k + 2);
        % Update x
        x = (1 - alpha) * x + alpha * s;
        history(k) = f(x);
        if norm(gradf(x)) < tol
            history = history(1:k);
            break;
        end
    end
end
