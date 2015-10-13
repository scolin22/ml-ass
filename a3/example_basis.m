
% Load data
load basisData.mat

degree = 8;
figure(1);

for d = 0:degree
    % Plot data
    subplot(3,3,d+1)
    plot(X,y,'b.')
    title(['degree = ', num2str(d)]);
    hold on

    % Fit least-squares estimator
    model = leastSquaresBasis(X,y,d);

    % Draw model prediction
    Xsample = [min(X):.1:max(X)]';
    yHat = model.predict(model,Xsample);
    plot(Xsample,yHat,'g-');
end
