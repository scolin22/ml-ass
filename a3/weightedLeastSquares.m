%% weightedLeastSquares
% Solve least squares problem with a weight for every training example.
% \ solves Ax = B for x
%
% We define $Z$ as a $n{\times}n$ diagonal matrix with weights along the
% diagonal.
%
% $\frac{1}{2}\sum\limits_{i=1}^n z_i(y_i-w^Tx_i)^2=\frac{1}{2}(y-Xw)^TZ(y-Xw)$
%
% $=\frac{1}{2}(y^T-w^TX^T)Z(y-Xw)$
%
% $=\frac{1}{2}(y^TZ-w^TX^TZ)(y-Xw)$
%
% $=\frac{1}{2}(y^TZy-y^TZXw-w^TX^TZy+w^TX^TZXw$
%
% Assuming $y^TZXw$ and $w^TX^TZy$ are scalar.
%
% $f(w)=\frac{1}{2}(y^TZy-2w^TX^TZy+w^TX^TZXw$
%
% ${\nabla}f(w)=X^TZy+X^TZXw=0$
%
% $w=(X^TZX)^{-1}X^TZy$
%
function [model] = weightedLeastSquares(X,y,z)
    z = [ones(1,400) 0.1*ones(1,100)];
    Z = diag(z);
    w = (X'*Z*X)\X'*Z*y;

    model.w = w;
    model.predict = @predict;
end

%% predict
function [yhat] = predict(model, Xtest)
    w = model.w;
    yhat = Xtest*w;
end
