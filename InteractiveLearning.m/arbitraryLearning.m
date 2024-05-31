function h = arbitraryLearning(H, teacher)
    % ARBITRARYLEARNING implements the Arbitrary Learning Algorithm.
    %   h = arbitraryLearning(H, teacher) performs arbitrary learning using
    %   the hypothesis class H and the teacher's responses.
    %
    % Inputs:
    %   H - hypothesis class (n x m matrix)
    %   teacher - function to query the teacher for counter-examples
    %
    % Outputs:
    %   h - learned hypothesis

    [~, m] = size(H);
    i = 1;
    while true
        % Pick an arbitrary hypothesis
        hi = arbitraryHypothesis(H, i);

        % Query the teacher
        xi = teacher(hi);

        % If no counter-example returned by teacher, output hypothesis
        if isempty(xi)
            h = hi;
            break;
        else
            % Eliminate hypotheses that disagree with teacher's response
            H = eliminateHypotheses(H, xi, hi(xi));
        end
        i = i + 1;
    end
end

function hi = arbitraryHypothesis(H, i)
    % ARBITRARYHYPOTHESIS picks an arbitrary hypothesis from the hypothesis class.
    %   hi = arbitraryHypothesis(H, i) picks an arbitrary hypothesis from the
    %   hypothesis class H based on index i.
    %
    % Inputs:
    %   H - hypothesis class (n x m matrix)
    %   i - index for picking hypothesis
    %
    % Outputs:
    %   hi - arbitrary hypothesis

    hi = H(i, :);
end
