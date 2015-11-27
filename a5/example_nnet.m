load nnetData.mat % Loads data {X,y}
% Standardize data
X = standardizeCols(X);

% Shuffle data
idx = randperm(length(X));
X = X(idx);
y = y(idx);

% Segment in to training and test data
[N,~] = size(X);
cv = N/2;

% 1 for Validation Error, 0 for Training Error
doVE = 1;
if doVE == 0
    Xv = X(cv+1:end,:);
    yv = y(cv+1:end,:);
    X = X(1:cv,:);
    y = y(1:cv,:);
else
    Xv = X;
    yv = y;
end;

% Add bias
[N,d] = size(X);
Xv = [ones(N,1) Xv];
X = [ones(N,1) X];
d = d + 1;

% Choose network structure
nNodes = 9;
nHidden = [nNodes];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-2;
beta = 1e-3;
funObj = @(w,i)MLPregressionLoss(w,X(i,:),y(i),nHidden);
sgdX = zeros(100,1);
sgdy = zeros(100,1);

% Derivative check that the gradient code is correct:
w0 = randn(nParams,1);
r = ceil(rand*N);
[f,g] = funObj(w0,r);
[f2,g2] = autoGrad(w0,@MLPregressionLoss,X(r,:),y(r),nHidden);
if max(abs(g-g2) > 1e-4)
    fprintf('User and numerical derivatives differ:\n');
    [g g2]
else
    fprintf('User and numerical derivatives agree.\n');
end

for bsiter = 1:10
    w = randn(nParams,1);
    wPrev = 0;
    sgdX = zeros(100,1);
    sgdy = zeros(100,1);
    for t = 1:maxIter
        % Record iteration vs "validation" error
        % if mod(t-1,round(maxIter/100)) == 0
        %     yhat = MLPregressionPredict(w,Xv,nHidden);
        %     sgdX((t-1)/round(maxIter/100)+1) = t;
        %     sgdy((t-1)/round(maxIter/100)+1) = sum((yv - yhat).^2);
        % end

        % The actual stochastic gradient algorithm:
        i = ceil(rand*N);
        [f,g] = funObj(w,i);
        wNew = w - stepSize*g + beta*(w - wPrev);
        if t > 1
            wPrev = w;
        end
        w = wNew;
    end
    % Plot "validation" error
    % figure(bsiter);clf;hold on
    % h=plot(sgdX(end/50:end),sgdy(end/50:end),'g-');
    % set(h,'LineWidth',3);
    % title(sprintf('iteration vs "validation" error for step size: %d', stepSize));
    % title(sprintf('iteration vs "validation" error for beta: %d', beta));
    % drawnow;

    % "validation" Error
    yhat = MLPregressionPredict(w,Xv,nHidden);
    valErr = sum((yv - yhat).^2);
    fprintf('Training iteration = %d, "validation" Error = %d\n',bsiter,valErr);

    fprintf('Training iteration = %d\n',t-1);
    figure(bsiter);clf;hold on
    Xhat = [-5:.05:5]';
    Xhat = [ones(size(Xhat,1),1) Xhat];
    yhat = MLPregressionPredict(w,Xhat,nHidden);
    plot(X(:,2),y,'.');
    h=plot(Xhat(:,2),yhat,'g-');
    set(h,'LineWidth',3);
    legend({'Data','Neural Net'});
    drawnow;

    % Run experiment with new step size
    % stepSize = stepSize*sqrt(10);

    % Run experiment with new momentum
    % beta = beta/sqrt(10);
end

