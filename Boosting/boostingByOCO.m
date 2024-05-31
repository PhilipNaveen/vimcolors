function [weights, models] = boostingByOCO(baseLearner, X_train, y_train, numRounds)
    % BOOSTINGBYOCO Boosting by Online Convex Optimization.
    %   [weights, models] = boostingByOCO(baseLearner, X_train, y_train, numRounds)
    %   performs boosting using online convex optimization with a given base learner.
    %
    % Inputs:
    %   baseLearner - a function handle for the base learning algorithm with signature @(X, y, weights)
    %   X_train - training data features
    %   y_train - training data labels
    %   numRounds - number of boosting rounds
    %
    % Outputs:
    %   weights - final weights for each base learner
    %   models - cell array of trained base learners
    
    n = size(X_train, 1);
    weights = ones(n, 1) / n; % Initialize weights
    models = cell(numRounds, 1);
    
    for t = 1:numRounds
        % Train base learner with current weights
        models{t} = baseLearner(X_train, y_train, weights);
        
        % Predict with base learner
        y_pred = predict(models{t}, X_train);
        
        % Compute weighted error
        epsilon_t = sum(weights .* (y_pred ~= y_train));
        
        % Compute alpha
        alpha_t = 0.5 * log((1 - epsilon_t) / epsilon_t);
        
        % Update weights
        weights = weights .* exp(-alpha_t * y_train .* y_pred);
        weights = weights / sum(weights); % Normalize weights
    end
end
