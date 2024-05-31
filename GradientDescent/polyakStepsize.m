function alpha = polyakStepsize(f, x, f_opt)
    % POLYAKSTEPSIZE Computes the Polyak stepsize for gradient descent.
    %   alpha = polyakStepsize(f, x, f_opt) computes the Polyak stepsize for
    %   the function f at point x given the optimal value f_opt.
    %
    % Inputs:
    %   f - function handle of the objective function
    %   x - current point
    %   f_opt - optimal value of the function
    %
    % Outputs:
    %   alpha - Polyak stepsize

    gradf = @(x) gradient(f, x);  % Compute gradient
    alpha = (f(x) - f_opt) / norm(gradf(x))^2;
end
