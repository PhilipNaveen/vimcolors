function [x, history] = approximatingLinearPrograms(A, b, c, x0, maxIter, tol)
    % APPROXIMATINGLINEARPROGRAMS Solves a linear programming problem approximately.
    %   [x, history] = approximatingLinearPrograms(A, b, c, x0, maxIter, tol) optimizes
    %   the function c'*x subject to A*x <= b using an iterative method starting from x0.
    %
    % Inputs:
    %   A - constraint matrix
    %   b - constraint vector
    %   c - cost vector
    %   x0 - initial guess
    %   maxIter - maximum number of iterations
    %   tol - tolerance for convergence
    %
    % Outputs:
    %   x - solution to the linear programming problem
    %   history - history of objective function values

    x = x0;
    history = zeros(maxIter, 1);
    
    for k = 1:maxIter
        % Subgradient calculation
        [~, idx] = max(A * x - b);
        g = A(idx, :)';
        
        % Step size
        alpha = 1 / sqrt(k);
        
        % Update x
        x = x - alpha * (g + c);
        
        % Projection onto feasible set
        x = projectionOntoFeasibleSet(x, A, b);
        
        % Objective value
        history(k) = c' * x;
        
        if norm(g) < tol
            history = history(1:k);
            break;
        end
    end
end

function x = projectionOntoFeasibleSet(x, A, b)
    % PROJECTONTOFEASIBLESET Projects x onto the feasible set defined by A * x <= b
    options = optimoptions('quadprog', 'Display', 'none');
    x = quadprog(eye(length(x)), -x, A, b, [], [], [], [], [], options);
end
