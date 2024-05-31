function [x, y, value] = zeroSumGame(A)
    % ZEROSUMGAME Solves a zero-sum game using linear programming.
    %   [x, y, value] = zeroSumGame(A) solves the zero-sum game defined by the matrix A.
    %
    % Inputs:
    %   A - payoff matrix
    %
    % Outputs:
    %   x - mixed strategy for player 1
    %   y - mixed strategy for player 2
    %   value - value of the game

    [m, n] = size(A);

    % Solve for player 1
    f = [-ones(m, 1); 1];
    A1 = [A', -ones(n, 1)];
    b1 = zeros(n, 1);
    Aeq1 = [ones(1, m), 0];
    beq1 = 1;
    lb1 = [zeros(m, 1); -Inf];
    x1 = linprog(f, A1, b1, Aeq1, beq1, lb1);

    % Solve for player 2
    g = [ones(n, 1); -1];
    A2 = [-A, ones(m, 1)];
    b2 = zeros(m, 1);
    Aeq2 = [ones(1, n), 0];
    beq2 = 1;
    lb2 = [zeros(n, 1); -Inf];
    y2 = linprog(g, A2, b2, Aeq2, beq2, lb2);

    x = x1(1:m);
    y = y2(1:n);
    value = x1(end);
end
