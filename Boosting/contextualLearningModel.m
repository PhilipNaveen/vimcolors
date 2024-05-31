function model = contextualLearningModel(baseLearner, contextFeature, X_train, y_train, numRounds)
    % CONTEXTUALLEARNINGMODEL Implements a contextual learning model using boosting.
    %   model = contextualLearningModel(baseLearner, contextFeature, X_train, y_train, numRounds)
    %   trains an ensemble of base learners using contextual information and boosting.
    %
    % Inputs:
    %   baseLearner - a function handle for the base learning algorithm with signature @(X, y, context)
    %   contextFeature - contextual feature vector
    %   X_train - training data features
    %   y_train - training data labels
    %   numRounds - number of boosting rounds
    %
    % Outputs:
    %   model - trained contextual learning model

    n = size(X_train, 1);
    weights = ones(n, 1) / n; % Initialize weights
    models = cell(numRounds, 1);
    alphas = zeros(numRounds, 1);
    
    for t = 1:numRounds
        % Train base learner with current weights and context
        models{t} = baseLearner(X_train, y_train, weights, contextFeature);
        
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
    
    % Aggregate models and alphas into a single model
    model.models = models;
    model.alphas = alphas;
    model.contextFeature = contextFeature;
end
