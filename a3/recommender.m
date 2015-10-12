%% Model
function [model] = recommender(X,K)
model.X = X;
model.K = K;
model.predict = @predict;
end

%% Predict
function [rec] = predict(model,j)
    [~,D] = size(model.X);
    % Find dot product
    dotP = model.X(:,j)' * model.X;
    % Find norms
    normP = norm(model.X(:,j)) * arrayfun(@(j) norm(model.X(:,j)), 1:D)';
    % Divide
    cosSim = bsxfun(@rdivide, dotP', normP);

    % Find k-largest, excluding first
    [~,idx] = sort(cosSim, 'descend');
    rec = idx(2:model.K+1,:);
end
