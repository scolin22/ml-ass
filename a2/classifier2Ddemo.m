% Load dataset
load binary.mat

[N,D] = size(X);

% Fit decision tree
model = decisionTree_InfoGain(X,y,1);

% Compute training error
yhat = model.predict(model,X);
trainingError = sum(yhat ~= y)/N

% Show data and decision boundaries
classifier2Dplot(X,y,model);