function value = vonNeumannTheorem(A)
    % VONNEUMANNTHEOREM Verifies the von Neumann Minimax Theorem.
    %   value = vonNeumannTheorem(A) returns the value of the game using von Neumann's theorem.
    %
    % Inputs:
    %   A - payoff matrix
    %
    % Outputs:
    %   value - value of the game
    
    [~, ~, value] = zeroSumGame(A);
end
