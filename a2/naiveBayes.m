function [model] = naiveBayes(X,y)
% [model] = naiveBayes(X,y,k)
%
% Implementation of navie Bayes classifier for binary features

% Compute number of training examples and number of features
[N,D] = size(X);

% Computer number of class lables
C = max(y);

counts = zeros(C,1);
for c = 1:C
    counts(c) = sum(y==c);
end
p_y = counts/N; % This is the probability of each class, p(y(i) = c)

% We will store:
%   p(x(i,j) = 1 | y(i) = c) as p_xy(j,1,c)
%   p(x(i,j) = 0 | y(i) = c) as p_xy(j,2,c)
p_xy = zeros(D,2,C);

for c = 1:C
    % Count of each feature with label c
    countDCTrue = sum(X(y==c,:)==1);
    countDCFalse = sum(X(y==c,:)==0);
    p_xy(:,1,c) = countDCTrue/counts(c);
    p_xy(:,2,c) = countDCFalse/counts(c);
end

model.C = C;
model.p_y = p_y;
model.p_xy = p_xy;
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
[T,D] = size(Xtest);
C = model.C;
p_y = model.p_y;
p_xy = model.p_xy;

yhat = zeros(T,1);
for i = 1:T
    probs = p_y; % This will be the probability for each class
    for j = 1:D
        if Xtest(i,j) == 1
            for c = 1:model.C
                probs(c) = probs(c)*p_xy(j,1,c);
            end
        else
            for c = 1:model.C
                probs(c) = probs(c)*p_xy(j,2,c);
            end
        end
    end
    [maxProb,yhat(i)] = max(probs);
end
end