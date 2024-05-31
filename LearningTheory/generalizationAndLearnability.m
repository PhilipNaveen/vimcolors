function [trainError, valError] = generalizationAndLearnability(model, X_train, y_train, X_val, y_val, complexity)
    % GENERALIZATIONANDLEARNABILITY Evaluates the generalization error of a model with different complexity levels.
    %   [trainError, valError] = generalizationAndLearnability(model, X_train, y_train, X_val, y_val, complexity)
    %   trains the model on the training data and computes the training and validation error for different complexity levels.
    %
    % Inputs:
    %   model - a function handle for the learning model with signature @(X, y, X_val, complexity)
    %   X_train - training data features
    %   y_train - training data labels
    %   X_val - validation data features
    %   y_val - validation data labels
    %   complexity - vector of complexity levels
    %
    % Outputs:
    %   trainError - training error for each complexity level
    %   valError - validation error for each complexity level

    trainError = zeros(length(complexity), 1);
    valError = zeros(length(complexity), 1);

    for i = 1:length(complexity)
        % Train the model with current complexity level
        y_train_pred = model(X_train, y_train, X_train, complexity(i));
        
        % Compute training error
        trainError(i) = mean(y_train_pred ~= y_train);
        
        % Predict on validation data
        y_val_pred = model(X_train, y_train, X_val, complexity(i));
        
        % Compute validation error
        valError(i) = mean(y_val_pred ~= y_val);
    end
end
