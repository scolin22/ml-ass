load cities.mat

X = ratings;
[n,d] = size(X);
k = 3;

ED = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*(X');

% Get all nodes' k-nearest-neighbours
[kMinDist, kMinIdx] = sort(ED);
nearestDist = kMinDist(2:k+1,:);
nearestNodes = kMinIdx(2:k+1,:);

% Average distance from node to neighbours
meanDist = mean(nearestDist);

% Average distance from neighbours to neighbours
neighbourDist = arrayfun(@(x) meanDist(x), nearestNodes);
meanNDist = mean(neighbourDist);

% Outlierness
outlierness = meanDist./meanNDist;

% Get top 10 outliers
[topOut, topIdx] = sort(outlierness, 'descend');

for col = 1:10
    fprintf('City: %s, Outlierness: %f\n', names(topIdx(:,col),:), topOut(:,col));
end
