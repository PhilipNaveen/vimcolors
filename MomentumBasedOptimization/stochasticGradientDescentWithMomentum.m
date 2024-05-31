function [x_opt, f_opt] = stochasticGradientDescentWithMomentum(f, grad_f, x0, options)
    % STOCHASTICGRADIENTDESCENTWITHMOMENTUM Implements the Stochastic Gradient Descent with Momentum optimization algorithm.
    %   [x_opt, f_opt] = stochasticGradientDescentWithMomentum(f, grad_f, x0, options) performs optimization of
    %   the objective function f using Stochastic Gradient Descent with Momentum algorithm.
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
    beta = 0.9;    % momentum coefficient
    maxIter = 1000;% maximum number of iterations

    % Override default options if provided
    if isfield(options, 'alpha')
        alpha = options.alpha;
    end
    if isfield(options, 'beta')
        beta = options.beta;
    end
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end

    % Initialize
    x = x0;
    v = zeros(size(x)); % momentum

    % Optimization loop
    for t = 1:maxIter
        % Sample a random data point
        rand_index = randi([1, length(x0)]);
        grad = grad_f(x(:, rand_index));

        % Update momentum
        v = beta * v + alpha * grad;

        % Update parameters
        x = x - v;
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
