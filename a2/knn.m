function [model] = knn(X,y,K)
% [model] = knn(X,y,k)
%
% Implementation of k-nearest neighbour classifer

model.X = X;
model.y = y;
model.K = K;
model.C = max(y);
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
X = model.X;
[N,D] = size(X);
[T,D] = size(Xtest);
yhat = zeros(T,1);

% create dist from model.X and Xtest
ED = X.^2*ones(D,T) + ones(N,D)*(Xtest').^2 - 2*X*(Xtest');

% get the model.K-least elements in dist and take the class label mode and insert in yhat
[kMinElems kMinIdx] = getNElements(ED, model.K);
kMinLabels = model.y(kMinIdx');
yhat = mode(kMinLabels,2);
end
