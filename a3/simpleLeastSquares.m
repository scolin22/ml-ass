%% simpleLeastSquares
% Includes bias variable $w_0$ and makes predictions using $y_i = wx_i + w_0$
% by adding column of 1's to X
function [model] = simpleLeastSquares(X, y)
    X = [ones(size(X, 1), 1) X];

    w = (X'*X)\X'*y;

    model.w = w;
    model.predict = @predict;
end

%% predict
function [yhat] = predict(model, Xtest)
    Xtest = [ones(size(Xtest, 1), 1) Xtest];

    w = model.w;
    yhat = Xtest*w;
end
