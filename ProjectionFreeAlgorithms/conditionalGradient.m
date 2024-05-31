function [x, history] = conditionalGradient(f, gradf, x0, A, b, maxIter, tol)
    % CONDITIONALGRADIENT Performs the conditional gradient (Frank-Wolfe) method.
    %   [x, history] = conditionalGradient(f, gradf, x0, A, b, maxIter, tol) optimizes
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

function s = linearOracle(gradient, A, b)
    % Solves the linear problem to find the update direction
    % min_s <gradient, s> subject to A*s <= b
    % For simplicity, we solve this using linprog (linear programming solver)
    
    K = length(b);
    f = gradient;
    options = optimoptions('linprog', 'Display', 'none');
    s = linprog(f, A, b, [], [], zeros(K, 1), [], options);
end
