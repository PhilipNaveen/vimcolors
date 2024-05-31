function distance = measureDistanceToOptimality(x, x_opt)
    % MEASUREDISTANCETOOPTIMALITY Measures the Euclidean distance to the optimal point.
    %   distance = measureDistanceToOptimality(x, x_opt) computes the Euclidean distance
    %   between the current point x and the optimal point x_opt.
    %
    % Inputs:
    %   x - current point
    %   x_opt - optimal point
    %
    % Outputs:
    %   distance - Euclidean distance to the optimal point

    distance = norm(x - x_opt);
end
