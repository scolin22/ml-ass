function [Z] = visualizeISOMAP(X,k,nb,names)

[n,d] = size(X);

% Compute all distances
D = X.^2*ones(d,n) + ones(n,d)*(X').^2 - 2*X*X';
D = sqrt(abs(D));

% Find nb-nearest-neighbours
[~,index] = sort(D);
for i = 1:n
    % Find indexes of nodes that aren't not the nearest neighbours
    notNB = setdiff(index(:,i),index(2:nb+1,i));
    % Set to zero
    D(notNB,i) = 0;
end

% Balance the adjacency matrix
for i = 1:n
    for j = i+1:n
        if (D(i,j) ~= D(j,i))
            if D(i,j)
                D(j,i) = D(i,j);
            else
                D(i,j) = D(j,i);
            end
        end
    end
end

% Find the shortest paths
for i = 1:n
    for j = i+1:n
        D(i,j) = dijkstra(D,i,j);
        D(j,i) = D(i,j);
    end
end

D(isinf(D)) =  max(D(isfinite(D)));

% Initialize low-dimensional representation with PCA
[U,S,V] = svd(X);
W = V(:,1:k)';
Z = X*W';

Z(:) = findMin(@stress,Z(:),500,0,D,names);

end

function [f,g] = stress(Z,D,names)

n = length(D);
k = numel(Z)/n;

Z = reshape(Z,[n k]);

f = 0;
g = zeros(n,k);
for i = 1:n
    for j = i+1:n
        % Objective Function
        Dz = norm(Z(i,:)-Z(j,:));
        s = D(i,j) - Dz;
        f = f + (1/2)*s^2;

        % Gradient
        df = s;
        dgi = (Z(i,:)-Z(j,:))/Dz;
        dgj = (Z(j,:)-Z(i,:))/Dz;
        g(i,:) = g(i,:) - df*dgi;
        g(j,:) = g(j,:) - df*dgj;
    end
end
g = g(:);

% Make plot if using 2D representation
if k == 2
    figure(3);
    clf;
    plot(Z(:,1),Z(:,2),'.');
    hold on;
    for i = 1:n
        text(Z(i,1),Z(i,2),names(i,:));
    end
    pause(.01)
end
end
