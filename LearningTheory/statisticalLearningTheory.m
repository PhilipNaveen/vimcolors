function [trainError, testError] = statisticalLearningTheory(model, X_train, y_train, X_test, y_test)
    % STATISTICALLEARNINGTHEORY Evaluates a model using training and testing data.
    %   [trainError, testError] = statisticalLearningTheory(model, X_train, y_train, X_test, y_test)
    %   trains the model on the training data and computes the training and testing error.
    %
    % Inputs:
    %   model - a function handle for the learning model with signature @(X, y, X_val)
    %   X_train - training data features
    %   y_train - training data labels
    %   X_test - testing data features
    %   y_test - testing data labels
    %
    % Outputs:
    %   trainError - training error
    %   testError - testing error

    % Train the model
    y_train_pred = model(X_train, y_train, X_train);
    
    % Compute training error
    trainError = mean(y_train_pred ~= y_train);
    
    % Predict on test data
    y_test_pred = model(X_train, y_train, X_test);
    
    % Compute testing error
    testError = mean(y_test_pred ~= y_test);
end
