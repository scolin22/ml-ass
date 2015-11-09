function [model] = logRegL1(X,y,lambda)

[n,d] = size(X);

maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm
w0 = rand(d,1);

% This is how you compute the function and gradient:
[f,g] = logisticLossL1(w0,X,y);
% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w0,@logisticLossL1,X,y);
if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

model.w = findMinL1(@logisticLossL1,w0,lambda,maxFunEvals,verbose,X,y);
model.predict = @(model,X)sign(X*model.w); % Predictions by taking sign
end

function [f,g] = logisticLossL1(w,X,y)
yXw = y.*(X*w);
f = sum(log(1+exp(-yXw))); % Function value
g = -X'*(y./(1+exp(yXw))); % Gradient
end
