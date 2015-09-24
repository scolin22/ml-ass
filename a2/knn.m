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
[T,D] = size(Xtest);
yhat = ones(T,1);
end