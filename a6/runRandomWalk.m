function [yhat] = runRandomWalk(Adj, labelList, i)
v = i;
while 1
% Check if v exists in labelList
neighbours = Adj(:,v);
if sum(labelList(:,1)==v)
    % Choose label with p = 1/(d_v + 1)
    if rand < 1/(sum(neighbours) + 1);
        break;
    end
end
% Set v to a neighbour, unbiased
v = randsample(find(neighbours), 1);
end
% Return the label
yhat = labelList(labelList(:,1)==v,2);
end
