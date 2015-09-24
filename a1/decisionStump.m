function [model] = decisionStump(X,y)
% [model] = decisionStump(X,y)
%
% Fits a decision stump that splits on a single variable,
%   assuming that X is binary {0,1}, and y is categorical {1,2,3,...,C}.

% Compute number of training examples and number of features [row, column]
% [posts, words] OR [objects, features]
[N,D] = size(X);

% Compute number of class labels
% Distinct newsgroups
C = max(y);

% Address the trivial case where we do not split
count = zeros(C,1);
for n = 1:N
    % Get number of instances of each category in y
    % Count instances of posts per newsgroups
    count(y(n)) = count(y(n)) + 1;
end
% Get the most popular instance by count
% Get newsgroup with the most posts
[maxCount,maxLabel] = max(count);
% See how many instances are not the most popular case
minError = sum(y ~= maxLabel);
splitVariable = [];
splitLabel0 = maxLabel;
splitLabel1 = [];

% Loop over features looking for the best split
if any(y ~= y(1))
    % Loop over all features
    % Loop over words
    for d = 1:D

        % Count number of class labels when the feature is 1, and when it is 0
        count1 = zeros(C,1);

        % Get appearance of particular word in all posts
        % Xvector = X(:,d);

        % Get convert to 1 or 0
        % Xvector1 = Xvector == 1;

        % Get indexes of these posts
        % Xvector1find = find(Xvector1);

        % Transpose
        % Xvector1findtran = Xvector1find';

        for n = find(X(:,d) == 1)'
            % For each post index, find what newsgroup it came from and increment the count of the newsgroup
            count1(y(n)) = count1(y(n)) + 1;
        end
        count0 = count-count1;

        % count1 is how many distinct words did each newsgroup mention (max is 100)
        % Compute majority class
        [maxCount,maxLabel1] = max(count1);
        [maxCount,maxLabel0] = max(count0);

        % Compute number of classification errors
        yhat = maxLabel0*ones(N,1);
        yhat(X(:,d) == 1) = maxLabel1;
        errors = sum(yhat~=y);

        % Compare to minimum error so far
        if errors < minError
            % This is the lowest error, store this value
            minError = errors;
            splitVariable = d;
            splitLabel1 = maxLabel1;
            splitLabel0 = maxLabel0;
        end
    end
end

model.splitVariable = splitVariable;
model.label1 = splitLabel1;
model.label0 = splitLabel0;
model.predictFunc = @predict;
end

function [y] = predict(model,X)
    [T,D] = size(X);

    if isempty(model.splitVariable)
        y = model.label0*ones(T,1);
    else
        y = zeros(T,1);
        for n = 1:T
            if X(n,model.splitVariable) == 1
                y(n,1) = model.label1;
            else
                y(n,1) = model.label0;
            end
        end
    end
end
