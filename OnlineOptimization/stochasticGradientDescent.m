function [x, history] = stochasticGradientDescent(f, gradf, x0, alpha, maxIter, batchSize)
    % STOCHASTICGRADIENTDESCENT Performs stochastic gradient descent optimization.
    %   [x, history] = stochasticGradientDescent(f, gradf, x0, alpha, maxIter, batchSize)
    %   optimizes the function f with its gradient gradf starting from x0 using step size alpha
    %   for maxIter iterations and batch size batchSize.
    %
    % Inputs:
    %   f - function handle of the objective function (sequence of functions)
    %   gradf - function handle of the gradient of f (sequence of gradients)
    %   x0 - initial guess
    %   alpha - step size
    %   maxIter - maximum number of iterations
    %   batchSize - size of the batch for stochastic updates
    %
    % Outputs:
    %   x - optimized parameters
    %   history - history of function values

    x = x0;
    history = zeros(maxIter, 1);
    n = numel(f);  % Total number of functions
    
    for k = 1:maxIter
        idx = randperm(n, batchSize);
        gradBatch = zeros(size(x));
        for j = 1:batchSize
            gradBatch = gradBatch + gradf(idx(j), x);
        end
        gradBatch = gradBatch / batchSize;
        x = x - alpha * gradBatch;
        history(k) = mean(arrayfun(@(i) f(i, x), idx));
    end
end
