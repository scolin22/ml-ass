function [model] = robustRegressionGradient(X,y,epsilon)

[n,d] = size(X);

% Initial guess
w0 = zeros(d,1);

% This is how you compute the function and gradient:
[f,g] = funObj(w0,X,y,epsilon);

% Derivative check that the gradient code is correct:
[f2,g2] = autoGrad(w0,@funObj,X,y,epsilon);

if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ (2nd is auto):\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

% Solve robust regression problem
w = findMin(@funObj,w0,100,X,y,epsilon);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
w = model.w;
yhat = Xtest*w;
end

function [f,g] = funObj(w,X,y,epsilon)
    u = (X*w-y).^2 + epsilon;
    f = sum(sqrt(u));

    v = 2*(X*w-y);
    g = sum(0.5*u.^-0.5.*v.*X);
end
