function [testingError] = doFold(trainData, trainLabels, testData, testLabels, depth)
    [N,D] = size(trainData);

    model = decisionTree_InfoGain(trainData,trainLabels,depth);

    % Generate a a prediction for y (yhat) from the testData
    yhat = model.predictFunc(model,testData);
    testingError = sum(yhat ~= testLabels)/N;
end
