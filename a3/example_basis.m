
% Load data
load basisData.mat

[N,~] = size(X);

Ntrain = N / 2;

Xtrain      = X(1:Ntrain,:);
Xvalidation = X(Ntrain+1:end,:);
ytrain      = y(1:Ntrain,:);
yvalidation = y(Ntrain+1:end,:);

degree = 20;

trainMSE = zeros(degree+1,1);
validationMSE = zeros(degree+1,1);

for d = 0:degree
    % Fit least-squares estimator
    model = leastSquaresBasis(Xtrain,ytrain,d);

    % Testing Error
    yhat = model.predict(model,Xtrain);
    trainMSE(d+1) = sum((yhat - ytrain).^2)/Ntrain;

    % Validation Error
    yhat = model.predict(model,Xvalidation);
    validationMSE(d+1) = sum((yhat - yvalidation).^2)/Ntrain;

    fprintf('degree %02d: training error: %f, validation error: %f\n',d,trainMSE(d+1),validationMSE(d+1));
end

% subplot(2,1,1);
% plot(0:degree, trainMSE, '-ro');
% legend('trainMSE');
% subplot(2,1,2);
plot(0:degree, abs(trainMSE - validationMSE)./trainMSE, '-ro');
legend('(trainMSE - validationMSE) / trainMSE');
