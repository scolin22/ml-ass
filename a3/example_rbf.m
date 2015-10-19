%% Load data
warning off all
close all
clear all
load nonLinearData.mat
[n,d] = size(X);

%% Find Sigma and Lambda with Lowest Error
Ntrain      = n/2;
Xtrain      = X(1:Ntrain,:);
Xvalidation = X(Ntrain+1:end,:);
ytrain      = y(1:Ntrain,:);
yvalidation = y(Ntrain+1:end,:);

maxVError  = inf;
vError     = inf;
bestSigma  = 0;
bestLambda = 0;

for sigma = 2.^[3:-1:-4]
    for lambda = 2.^[-12:2]
        % Train on X, test o Xtest
        model  = leastSquaresRBF(Xtrain,ytrain,sigma,lambda);
        yhat   = model.predict(model,Xvalidation);
        vError = mean(abs(yhat-yvalidation));
        % fprintf('Test error with sigma = %f lambda = %f is %f\n',sigma,lambda,vError);
        if (vError < maxVError)
            maxVError  = vError;
            bestSigma  = sigma;
            bestLambda = lambda;
        end
    end
end

sigma  = bestSigma;
lambda = bestLambda;

%% Plotting Code
plot(X,y,'b.');hold on
plot(Xtest,ytest,'g.');
xl = xlim;
yl = ylim;
Xvals = [xl(1):.1:xl(2)]';
pause(.1)

% Display result of fitting with RBF kernel
% Train on X, test on Xtest
model = leastSquaresRBF(X,y,sigma,lambda);
yhat = model.predict(model,Xtest);
fprintf('Test error with sigma = %f lambda = %f is %f\n',sigma,lambda,mean(abs(yhat-ytest)));

figure(1);clf;
plot(X,y,'b.');hold on
plot(Xtest,ytest,'g.');
yvals = model.predict(model,Xvals);
plot(Xvals,yvals,'r-');
legend({'Train','Test'});
ylim(yl);
title(sprintf('RBF Basis (sigma = %f)',sigma));
