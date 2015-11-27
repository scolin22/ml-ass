function [model] = recommendSVD(X,y,k)

n = max(X(:,1));
d = max(X(:,2));
nRatings = size(X,1);

% Initialize parameters
% - for the biases, we'll use the user/item averages
% - for the latent factors, we'll use small random values
subModel = recommendUserItemMean(X,y);
bu = subModel.bu/2;
bm = subModel.bm/2;
W = .00001*randn(k,d);
Z = .00001*randn(n,k);

% Optimization
maxIter = 10;
alpha = 1e-2;
for iter = 1:maxIter*nRatings

    % Select random example
    i = randi([1 nRatings]);

    % Make prediction for this rating based on current model
    u = X(i,1);
    m = X(i,2);
    yhat = bu(u) + bm(m) + W(:,m)'*Z(u,:)';

    % Calculate gradient of this prediction of the random example
    % (follows from chain rule)
    r = y(i)-yhat;

    % Take a small step in the negative gradient directions
    bu(u) = bu(u) - alpha*-r;
    bm(m) = bm(m) - alpha*-r;
    W(:,m) = W(:,m) - alpha*-r*Z(u,:)';
    Z(u,:) = Z(u,:) - alpha*-r*W(:,m)';
end

model.bu = bu;
model.bm = bm;
model.W = W;
model.Z = Z;
model.predict = @predict;
end

function [y] = predict(model,X)
t = size(X,1);
bu = model.bu;
bm = model.bm;
W = model.W;
Z = model.Z;

y = zeros(t,1);
for i = 1:t
    u = X(i,1);
    m = X(i,2);
    y(i) = bu(u) + bm(m) + W(:,m)'*Z(u,:)'; % Take the average between user and movie ratings
end
end
