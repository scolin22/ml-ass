
% Load data
load outliersData.mat

% Plot data
figure(1);
plot(X,y,'b.')
title('Training Data');
hold on

% Fit least-squares estimator
z = [ones(1,400) 0.1*ones(1,100)];
model = weightedLeastSquares(X,y,z);

% Draw model prediction
Xsample = [min(X):.01:max(X)]';
yHat = model.predict(model,Xsample);
plot(Xsample,yHat,'g-');
