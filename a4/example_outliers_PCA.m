load cities.mat

X = ratings;
[n,d] = size(X);
k = 2;
X = standardizeCols(X);

[U,S,V] = svd(X);
W = V(:,1:k)';
% Check L2 norm of each row is 1
% and Inner Product between first and all other rows is 0
for row = 1:k
    if norm(W(k,:)) - 1 > 1e-10
        fprintf('Scaling issue\n');
    end
    if W(1,:)*W(k,:)' > 1e-10
        fprintf('Orthogonality issue\n');
    end
end

Z = X*W';

figure(1);
plot(Z(:,1),Z(:,2),'.');
gname(names);
