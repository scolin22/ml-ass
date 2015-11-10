load animals.mat

[n,d] = size(X);
X = standardizeCols(X);

[U,S,V] = svd(X);

Var = S(1:n,1:n).^2;
Var_sum = trace(Var);
Var_norm = S2./Var_sum;

X_fro = norm(X,'fro')^2;
for k = 1:d
    W = V(:,1:k)';
    for row = 1:k
    % Check L2 norm of each row is 1
    % and Inner Product between first and all other rows is 0
        if norm(W(k,:)) - 1 > 1e-10
            fprintf('Scaling issue\n');
        end
        if W(1,:)*W(k,:)' > 1e-10 && 1 ~= k
            fprintf('Orthogonality issue\n');
        end
    end
    Z = X*W';
    numer_fro = norm(X-Z*W,'fro')^2;
    if k < n
        % Normalized cumulative sum of the variance along the diagonal
        Var_ncsum = abs(1 - trace(Var_norm(1:k,1:k)));
    end
    fprintf('k: %d, Compression Ratio: %f, 1-VarNCSum: %f\n', k, numer_fro/X_fro, Var_ncsum);
end
