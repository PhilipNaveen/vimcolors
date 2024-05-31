function [primal, dual] = linearProgrammingDuality(c, A, b)
    % LINEARPROGRAMMINGDUALITY Solves the primal and dual linear programming problems.
    %   [primal, dual] = linearProgrammingDuality(c, A, b) solves the primal problem
    %   min c'*x subject to A*x <= b and its dual max b'*y subject to A'*y = c, y >= 0.
    %
    % Inputs:
    %   c - cost vector
    %   A - constraint matrix
    %   b - constraint vector
    %
    % Outputs:
    %   primal - solution to the primal problem
    %   dual - solution to the dual problem
    
    options = optimoptions('linprog', 'Display', 'none');
    [primal, ~, ~, ~, lambda] = linprog(c, A, b, [], [], [], [], options);
    dual = lambda.ineqlin;
end
