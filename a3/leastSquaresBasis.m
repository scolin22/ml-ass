%% leastSquaresBasis
% Includes bias variable $w_0$ and makes predictions using $y_i = wx_i + w_0$
% by adding column of 1's to X
function [model] = leastSquaresBasis(X,y,d)
    X = polyBasis(X,d);

    w = (X'*X)\X'*y;

    model.w = w;
    model.d = d;
    model.predict = @predict;
end

%% predict
function [yhat] = predict(model, Xtest)
    Xtest = polyBasis(Xtest, model.d);

    yhat = Xtest*model.w;
end

%% Transform X into Xpoly
function [Xpoly] = polyBasis(X,d)
    [N,~] = size(X);
    Xpoly = bsxfun(@times, ones(N, d+1), X);
    Xpoly = bsxfun(@power, Xpoly, 0:d);
end
