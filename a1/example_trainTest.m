clear all
load DTdata.mat

T = length(ytest);
validationError = zeros(15,1);
firstFold = X(1:2500,:);
secondFold = X(2501:5000,:);
firstLabels = y(1:2500,:);
secondLabels = y(2501:5000,:);
for depth = 1:15
    % subset into 1:2500 and 2501:5000
    % average errors
    % record into output
    firstTestingError = doFold(firstFold, firstLabels, secondFold, secondLabels, depth);
    secondTestingError = doFold(secondFold, secondLabels, firstFold, firstLabels, depth);
    validationError(depth) = (firstTestingError + secondTestingError) / 2;
end

scatter(1:15, validationError);
