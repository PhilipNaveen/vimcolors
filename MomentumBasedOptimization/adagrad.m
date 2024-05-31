function [x_opt, f_opt] = adagrad(f, grad_f, x0, options)
    % ADAGRAD Implements the Adagrad optimization algorithm.
    %   [x_opt, f_opt] = adagrad(f, grad_f, x0, options) performs optimization of
    %   the objective function f using Adagrad algorithm.
    %
    % Inputs:
    %   f - objective function
    %   grad_f - gradient of the objective function
    %   x0 - initial point
    %   options - optimization options (optional)
    %
    % Outputs:
    %   x_opt - optimal point
    %   f_opt - optimal function value

    if nargin < 4
        options = struct();
    end

    % Default hyperparameters
    alpha = 0.01;  % learning rate
    epsilon = 1e-8;% small constant for numerical stability
    maxIter = 1000;% maximum number of iterations

    % Override default options if provided
    if isfield(options, 'alpha')
        alpha = options.alpha;
    end
    if isfield(options, 'epsilon')
        epsilon = options.epsilon;
    end
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end

    % Initialize
    x = x0;
    G = zeros(size(x)); % sum of squares of past gradients

    % Optimization loop
    for t = 1:maxIter
        % Compute gradient
        grad = grad_f(x);

        % Accumulate squared gradient
        G = G + grad.^2;

        % Update parameters
        x = x - alpha * grad ./ (sqrt(G) + epsilon);
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
