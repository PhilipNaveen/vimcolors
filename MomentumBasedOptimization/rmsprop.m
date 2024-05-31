function [x_opt, f_opt] = rmsprop(f, grad_f, x0, options)
    % RMSPROP Implements the RMSProp optimization algorithm.
    %   [x_opt, f_opt] = rmsprop(f, grad_f, x0, options) performs optimization of
    %   the objective function f using RMSProp algorithm.
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
    alpha = 0.001; % learning rate
    beta = 0.9;    % decay rate
    epsilon = 1e-8;% small constant for numerical stability
    maxIter = 1000;% maximum number of iterations

    % Override default options if provided
    if isfield(options, 'alpha')
        alpha = options.alpha;
    end
    if isfield(options, 'beta')
        beta = options.beta;
    end
    if isfield(options, 'epsilon')
        epsilon = options.epsilon;
    end
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end

    % Initialize
    x = x0;
    E_g = zeros(size(x)); % running average of squared gradients

    % Optimization loop
    for t = 1:maxIter
        % Compute gradient
        grad = grad_f(x);

        % Compute squared gradient
        E_g = beta * E_g + (1 - beta) * grad.^2;

        % Update parameters
        x = x - alpha * grad ./ (sqrt(E_g) + epsilon);
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
