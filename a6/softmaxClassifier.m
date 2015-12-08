function [model] = softmaxClassifier(X,y)
% Classification using one-vs-all least squares

% Compute sizes
[n,d] = size(X);
k = max(y);

w = randn(d,k);
% This is how you compute the function and gradient:
[f,g] = softmaxLoss(w(:),X,y,k);
% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w(:),@softmaxLoss,X,y,k);
if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

w = zeros(d,k);
verbose = 0;
w(:) = findMin(@softmaxLoss,w(:),200000,verbose,X,y,k);
model.W = w;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [f,g] = softmaxLoss(w,X,y,k)
[n,d] = size(X);

w = reshape(w, [d k]);

p_i = exp(X*w);
prob = zeros(n,1);
for i = 1:n
    p_i(i,:) = p_i(i,:)./sum(p_i(i,:)); % get e^(x_i*w_k), i.e. softmax

    c = y(i);
    prob(i) = p_i(i,c); % select p_i of the correct class
end

f = -sum(log(prob));

grad = zeros(d,k);
for i = 1:n
    bin_sel = zeros(1,k);
    c = y(i);
    bin_sel(c) = 1;
    grad = grad + -(X(i,:)'*(bin_sel-p_i(i,:)));
end

g = reshape(grad, [d*k 1]);
end
