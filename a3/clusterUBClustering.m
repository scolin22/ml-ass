function [model] = clusterUBClustering(X,K,nModels)
[N,D] = size(X);

for m = 1:nModels
    model.subModel{m} = clusterKmeans(X,K);
end

for m = 1:nModels
    clusters(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end

% New Code
% Store likelihood of pairs appearing in the same cluster
pairProb = zeros(N, N);

for m = 1:nModels
    for k = 1:K
        same = clusters(:, m) == k;
        idx = find(same);

        % Sum new occurences of nodes appearing the same cluster
        pairProb(:, idx) = pairProb(:, idx) + repmat(same, size(idx'));
    end
end

% Calculate probabilities
pairProb = pairProb * 1/nModels;

% Cluster based on probabilities of pairs appearing in the same cluster > 0.5
clusters = clusterProbBasedcluster(X, 0.5, pairProb);
% End New Code

model.clusters = clusters;
end

%% Probablistic Based Clustering
function [cluster] = clusterProbBasedcluster(X,eps,D)
[N,~] = size(X);

% This will be the cluster of each object.
cluster = zeros(N,1);

% This variable will keep track of whether we've visited each object.
visited = zeros(N,1);

% K will count the number of clusters we've found
K = 0;
for i = 1:N
    if ~visited(i)
        % We only need to consider examples that have never been visited
        visited(i) = 1;
        % Take likelihoods > eps
        neighbors = find(D(:,i) > eps);
        if length(neighbors) > 0
            % We found a new cluster
            K = K + 1;
            [visited,cluster] = expand(X,i,neighbors,K,eps,D,visited,cluster);
        end
    end
end
end

function [visited,cluster] = expand(X,i,neighbors,K,eps,D,visited,cluster)
cluster(i) = K;
ind = 0;
while 1
    ind = ind+1;
    if ind > length(neighbors)
        break;
    end
    n = neighbors(ind);
    cluster(n) = K;

    if ~visited(n)
        visited(n) = 1;
        % Take likelihoods > eps
        neighbors2 = find(D(:,n) > eps);
        neighbors = [neighbors;setdiff(neighbors2,neighbors)];
    end
end
end
