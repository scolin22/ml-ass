% Load X and y variable
load newsgroups.mat
[N,D] = size(X);

% Fit decision stump
model = decisionTree(X,y,2);

% Evaluate training error
yhat = model.predictFunc(model,X);
error = sum(yhat ~= y)/N;
fprintf('Error with decision stump: %.2f\n',error);
