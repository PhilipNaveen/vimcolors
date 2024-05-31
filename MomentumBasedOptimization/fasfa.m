function [x_opt, f_opt] = fasfa(f, grad_f, x0, options)
    % FASFA Implements the Fast Adaptive Stochastic Function Accelerator (FASFA) algorithm.
    %   [x_opt, f_opt] = fasfa(f, grad_f, x0, options) performs optimization of
    %   the objective function f using the FASFA algorithm.
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
    alpha = 0.001;  % learning rate
    mu = 0.8;        % first order momentum decay estimate
    nu = 0.999;      % second order momentum decay estimate
    epsilon = 1e-8;  % small constant for numerical stability
    maxIter = 1000;  % maximum number of iterations

    % Override default options if provided
    if isfield(options, 'alpha')
        alpha = options.alpha;
    end
    if isfield(options, 'mu')
        mu = options.mu;
    end
    if isfield(options, 'nu')
        nu = options.nu;
    end
    if isfield(options, 'epsilon')
        epsilon = options.epsilon;
    end
    if isfield(options, 'maxIter')
        maxIter = options.maxIter;
    end

    % Initialize
    x = x0;
    m = zeros(size(x)); % first moment vector
    n = zeros(size(x)); % second moment vector
    t = 0; % timestep/iteration

    % Optimization loop
    while t < maxIter
        t = t + 1;

        % Compute gradient
        grad = grad_f(x);

        % Update biased first moment estimation
        m = mu * m + (1 - mu) * grad;

        % Update biased second moment estimation
        n = nu * n + (1 - nu) * (grad.^2);

        % Compute raw moment estimates
        m_hat = m / (1 - mu^t);
        n_hat = n / (1 - nu^t);

        % Implement FASFA update rule
        x = x - alpha * m_hat ./ (sqrt(n_hat) + epsilon);
    end

    % Compute optimal function value
    f_opt = f(x);
    x_opt = x;
end
