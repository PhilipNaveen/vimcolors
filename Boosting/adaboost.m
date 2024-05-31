function [models, alphas] = adaboost(baseLearner, X_train, y_train, numRounds)
    % ADABOOST Implements the AdaBoost algorithm.
    %   [models, alphas] = adaboost(baseLearner, X_train, y_train, numRounds)
    %   trains an ensemble of base learners using the AdaBoost algorithm.
    %
    % Inputs:
    %   baseLearner - a function handle for the base learning algorithm with signature @(X, y, weights)
    %   X_train - training data features
    %   y_train - training data labels
    %   numRounds - number of boosting rounds
    %
    % Outputs:
    %   models - cell array of trained base learners
    %   alphas - weights for each base learner

    n = size(X_train, 1);
    weights = ones(n, 1) / n; % Initialize weights
    models = cell(numRounds, 1);
    alphas = zeros(numRounds, 1);
    
    for t = 1:numRounds
        % Train base learner with current weights
        models{t} = baseLearner(X_train, y_train, weights);
        
        % Predict with base learner
        y_pred = predict(models{t}, X_train);
        
        % Compute weighted error
        epsilon_t = sum(weights .* (y_pred ~= y_train));
        
        % Compute alpha
        alpha_t = 0.5 * log((1 - epsilon_t) / epsilon_t);
        alphas(t) = alpha_t;
        
        % Update weights
        weights = weights .* exp(-alpha_t * y_train .* y_pred);
        weights = weights / sum(weights); % Normalize weights
    end
end
