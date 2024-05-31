function h = majorityLearning(H, teacher)
    % MAJORITYLEARNING implements the Majority Learning Algorithm.
    %   h = majorityLearning(H, teacher) performs majority learning using
    %   the hypothesis class H and the teacher's responses.
    %
    % Inputs:
    %   H - hypothesis class (n x m matrix)
    %   teacher - function to query the teacher for counter-examples
    %
    % Outputs:
    %   h - learned hypothesis

    [~, m] = size(H);
    while true
        % Pick the majority hypothesis
        h = majorityHypothesis(H);

        % Query the teacher
        x = teacher(h);

        % If no counter-example returned by teacher, output hypothesis
        if isempty(x)
            break;
        else
            % Eliminate hypotheses that disagree with teacher's response
            H = eliminateHypotheses(H, x, h(x));
        end
    end
end

function h = majorityHypothesis(H)
    % MAJORITYHYPOTHESIS computes the majority hypothesis.
    %   h = majorityHypothesis(H) computes the majority hypothesis by
    %   setting the value in each column to the most frequent element.
    %   Ties are broken in favor of the smaller element.
    %
    % Inputs:
    %   H - hypothesis class (n x m matrix)
    %
    % Outputs:
    %   h - majority hypothesis

    [~, m] = size(H);
    h = zeros(1, m);
    for j = 1:m
        h(j) = mode(H(:, j));
    end
end

function H = eliminateHypotheses(H, x, label)
    % ELIMINATEHYPOTHESES eliminates hypotheses that disagree with
    % the teacher's response on the given counter-example.
    %   H = eliminateHypotheses(H, x, label) removes hypotheses from H
    %   that disagree with the given label on the column x.
    %
    % Inputs:
    %   H - hypothesis class (n x m matrix)
    %   x - column index of counter-example
    %   label - label provided by the teacher for the counter-example

    idx = H(:, x) ~= label;
    H(idx, :) = [];
end
