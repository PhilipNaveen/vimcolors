function [x_opt, f_opt] = nestrovMomentum(f, grad_f, x0, options)
    % NESTROVMOMENTUM Implements the Nesterov Momentum optimization algorithm.
    %   [x_opt, f_opt] = nestrovMomentum(f, grad_f, x0, options) performs optimization of
    %   the objective function f using Nesterov Momentum algorithm.
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
    v = zeros(size(x)); % velocity

    % Optimization loop
    for t = 1:maxIter
        % Compute gradient at lookahead point
        lookahead_x = x - beta * v;
        grad = grad_f(lookahead_x);

        % Update velocity
        v = beta * v + alpha * grad;

        % Update parameters
        x = x - v;
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
