function [w, b, history] = svmTraining(X, y, lambda, maxIter, alpha)
    % SVMTRAINING Trains a Support Vector Machine using gradient descent.
    %   [w, b, history] = svmTraining(X, y, lambda, maxIter, alpha) trains an SVM
    %   with regularization parameter lambda using gradient descent for maxIter iterations
    %   and step size alpha.
    %
    % Inputs:
    %   X - matrix of training examples (each row is a training example)
    %   y - vector of labels (+1 or -1)
    %   lambda - regularization parameter
    %   maxIter - maximum number of iterations
    %   alpha - step size
    %
    % Outputs:
    %   w - weight vector
    %   b - bias term
    %   history - history of loss values

    [m, n] = size(X);
    w = zeros(n, 1);
    b = 0;
    history = zeros(maxIter, 1);

    for iter = 1:maxIter
        for i = 1:m
            if y(i) * (X(i, :) * w + b) < 1
                w = w - alpha * (lambda * w - y(i) * X(i, :)');
                b = b + alpha * y(i);
            else
                w = w - alpha * lambda * w;
            end
        end
        % Compute hinge loss
        loss = sum(max(0, 1 - y .* (X * w + b))) / m + (lambda / 2) * norm(w)^2;
        history(iter) = loss;
    end
end
