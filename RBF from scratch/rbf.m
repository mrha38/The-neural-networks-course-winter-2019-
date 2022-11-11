clc
clear all
dataSet = importdata('iris.mat');

%Initial importante variables 
k = 8;
[idx, centers] = kmeans(dataSet.X, k);
spreed = 1;
trainP = 0.7;
testP = 0.3;
[sizeOfDataSetX, ~] = size(dataSet.X);

%Normalize Data
dataSet.X = normalize(dataSet.X);

%Defile Train and Test Matrices, Create Target matrix for Train and Test
randP = randperm(sizeOfDataSetX);

indexsForTrian = randP(1:trainP * sizeOfDataSetX);
trainData = dataSet.X(indexsForTrian(:),:);
[~, sizeOfindexsForTrian] = size(indexsForTrian);
trainDataClassZ = dataSet.class(indexsForTrian(:));
[sizeOfTrainDataClass, ~] = size(trainDataClassZ);
trainDataClass = zeros(sizeOfTrainDataClass, 3);

for i = 1:sizeOfTrainDataClass
    if trainDataClassZ(i) == 1
        trainDataClass(i, 1) = 1;
    elseif trainDataClassZ(i) == 2
        trainDataClass(i, 2) = 1;
    else
        trainDataClass(i, 3) = 1;
    end
end

indexsForTest = randP(trainP * sizeOfDataSetX + 1:sizeOfDataSetX);
testData = dataSet.X(indexsForTest(:),:);
testDataClassZ = dataSet.class(indexsForTest(:));
[sizeOfTestDataClass, ~] = size(testDataClassZ);
testDataClass = zeros(sizeOfTestDataClass, 3);
for i = 1:sizeOfTestDataClass
    if testDataClassZ(i) == 1
        testDataClass(i, 1) = 1;
    elseif testDataClassZ(i) == 2
        testDataClass(i, 2) = 1;
    else
        testDataClass(i, 3) = 1;
    end
end

%Compute Weight matrix
distanceMatrix = exp(-dist(trainData, centers').^2 ./ (2 * spreed^2));
weightMatrix = mtimes(pinv(distanceMatrix), trainDataClass);

%RESULT
distanceMatrix = exp(-dist(testData, centers').^2 / (2 * spreed^2));
result = mtimes(distanceMatrix, weightMatrix);
[sizeOfResult, ~] = size(result);

resultClass = zeros(sizeOfResult, 3);
for i=1:sizeOfResult
    [~, indexOfMaxInRow] = max(result(i, :));
    resultClass(i, indexOfMaxInRow) = 1;
end

predict = sign(testDataClass - resultClass);

numberOfSuccessPredict = 0;
numberOfFaultPredict = 0;

for i=1:sizeOfResult
    if predict(i, 1) == 0 && predict(i, 2) == 0 && predict(i, 3) == 0
        numberOfSuccessPredict = numberOfSuccessPredict + 1;
    else
        numberOfFaultPredict = numberOfFaultPredict + 1;
    end
end

disp('Number of Fault preict:');
disp(numberOfFaultPredict);
disp('Number of Success preict:');
disp(numberOfSuccessPredict);
disp('accuracy:');
disp((numberOfSuccessPredict - numberOfFaultPredict) / sizeOfResult * 100);
