function [model] = clusterKmeans(X,K)
% [model] = clusterKmeans(X,K)
%
% K-means clustering

[N,D] = size(X);

% Choose random points to initialize means
% means = zeros(K,D);
% for k = 1:K
%     i = ceil(rand*N);
%     means(k,:) = X(i,:);
% end

% Instead of k random points
% start with 1 random point, C, from X
% Compute EuclDist from all points to C
% chose another point based on the EuclDist^2 distribution, chose the furthest disance
% EuclDist^2 determined by the closest mean

means = zeros(K,D);
ED = 1/N*ones(N,1);
for k = 1:K
    OldED = ED;
    means(k,:) = X(sampleDiscrete(OldED),:);

    ED = sqrt(X.^2*ones(D,K) + ones(N,D)*(means').^2 - 2*X*means');
    minED = min(ED,[],2);
    totalED = sum(minED(:,1:k),1);
end

X2 = X.^2*ones(D,K);
while 1
    means_old = means;

    % Compute Euclidean distance between each data point and each mean
    distances = sqrt(X2 + ones(N,D)*(means').^2 - 2*X*means');

    % Assign each data point to closest mean
    [~,clusters] = min(distances,[],2);

    % Compute mean of each cluster
    means = zeros(K,D);
    for k = 1:K
        means(k,:) = mean(X(clusters==k,:),1);
    end

    % If we only have two features, make a colored scatterplot
    if D == 2
        clf;hold on;
        colors = getColors;
        for k = 1:K
            h = plot(X(clusters==k,1),X(clusters==k,2),'.');
            set(h,'Color',colors{k});
        end
        pause(.25);
    end

    fprintf('Running K-means, difference = %f\n',max(max(abs(means-means_old))));

    if max(max(abs(means-means_old))) < 1e-5
        break;
    end
end

model.means = means;
model.clusters = clusters;
