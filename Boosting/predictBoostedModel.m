function y_pred = predictBoostedModel(model, X)
    % PREDICTBOOSTEDMODEL Predicts using a boosted model.
    %   y_pred = predictBoostedModel(model, X) predicts the labels for input data X using
    %   the provided boosted model.
    %
    % Inputs:
    %   model - a struct containing the trained boosted model with fields 'models' and 'alphas'
    %   X - input data features
    %
    % Outputs:
    %   y_pred - predicted labels

    numRounds = length(model.models);
    aggregatedPredictions = zeros(size(X, 1), 1);
    
    for t = 1:numRounds
        % Predict with each base learner and aggregate
        y_pred_t = predict(model.models{t}, X);
        aggregatedPredictions = aggregatedPredictions + model.alphas(t) * y_pred_t;
    end
    
    % Final prediction by taking the sign of the aggregated predictions
    y_pred = sign(aggregatedPredictions);
end
