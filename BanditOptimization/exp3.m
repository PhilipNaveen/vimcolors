function [weights, history] = exp3(T, K, gamma)
    % EXP3 Performs the EXP3 algorithm for the multi-armed bandit problem.
    %   [weights, history] = exp3(T, K, gamma) runs the EXP3 algorithm for T rounds with
    %   K arms and exploration parameter gamma.
    %
    % Inputs:
    %   T - number of rounds
    %   K - number of arms
    %   gamma - exploration parameter
    %
    % Outputs:
    %   weights - final weights of the arms
    %   history - history of chosen arms and rewards

    weights = ones(K, 1);
    history = zeros(T, 2); % Column 1: chosen arm, Column 2: received reward
    
    for t = 1:T
        probs = (1 - gamma) * (weights / sum(weights)) + (gamma / K);
        arm = randsample(1:K, 1, true, probs);
        reward = banditReward(arm, t);
        estimatedReward = reward / probs(arm);
        
        % Update weights
        weights(arm) = weights(arm) * exp(gamma * estimatedReward / K);
        
        % Store history
        history(t, :) = [arm, reward];
    end
end



% OVERRIDE if wanted to
function reward = banditReward(arm, t)
    % Simulated reward function for the chosen arm
    % For the sake of example, let's assume a random reward mechanism
    reward = rand;
end
