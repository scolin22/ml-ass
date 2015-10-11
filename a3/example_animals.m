%% Animals with attributes data
load animals.mat

%% Density-Based Clustering
eps = 13;
minPts = 3;
model = clusterDBcluster(X,eps,minPts);

%% Print Clusters
for k = 1:max(model.clusters)
    fprintf('Cluster %d: ',k);
    fprintf('%s ',animals{model.clusters==k});
    fprintf('\n');
end
