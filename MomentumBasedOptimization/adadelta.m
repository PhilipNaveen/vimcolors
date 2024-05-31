function [x_opt, f_opt] = adadelta(f, grad_f, x0, options)
    % ADADELTA Implements the AdaDelta optimization algorithm.
    %   [x_opt, f_opt] = adadelta(f, grad_f, x0, options) performs optimization of
    %   the objective function f using AdaDelta algorithm.
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
    rho = 0.9;     % decay rate
    epsilon = 1e-8;% small constant for numerical stability
    maxIter = 1000;% maximum number of iterations

    % Override default options if provided
    if isfield(options, 'rho')
        rho = options.rho;
    end
    if isfield(options, 'epsilon')
        epsilon = options.epsilon;
    end
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end

    % Initialize
    x = x0;
    E_delta_x = zeros(size(x)); % accumulated squared deltas
    E_delta_x_t = zeros(size(x)); % accumulated squared deltas (time varying)

    % Optimization loop
    for t = 1:maxIter
        % Compute gradient
        grad = grad_f(x);

        % Compute squared gradient
        grad_sq = grad.^2;

        % Compute exponentially decaying average of squared deltas
        E_delta_x = rho * E_delta_x + (1 - rho) * grad_sq;

        % Compute RMS of squared deltas
        RMS_delta_x = sqrt(E_delta_x + epsilon);

        % Compute RMS of squared deltas (time varying)
        RMS_delta_x_t = sqrt(E_delta_x_t + epsilon);

        % Update parameters
        delta_x = - (RMS_delta_x_t ./ RMS_delta_x) .* grad;
        x = x + delta_x;

        % Compute exponentially decaying average of squared deltas (time varying)
        E_delta_x_t = rho * E_delta_x_t + (1 - rho) * delta_x.^2;
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
