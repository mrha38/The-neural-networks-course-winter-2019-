clc
clear all
%Read DataSet
dataSet = importdata('iris.mat');

%Initial importante variables
epoc = 50;
LRMax = 0.5;
LRMin = 0.09;
SOMSizeRows = 4;
SOMSizeColumns = 4;
[sizeOfDataSetX, sizeOfDataSetY] = size(dataSet.X);
trainP = 0.7;
testP = 0.3;
randP = randperm(sizeOfDataSetX);

%Normalize DataSet
dataSet.X = normalize(dataSet.X);

%Defile SOM Network
SOMNetwork = randn(SOMSizeRows, SOMSizeColumns, sizeOfDataSetY);

%Coordinates
coordinates = zeros(2, SOMSizeRows * SOMSizeColumns);
[sizeOfCoordinatesX, sizeOfCoordinatesY] = size(coordinates);
index = 1;
for i = 1:SOMSizeRows
    for j = 1:SOMSizeColumns
        coordinates(1, index) = i;
        coordinates(2, index) = j;
        index = index + 1;
    end
end

%Distance
distance = mandist(coordinates);
spreed = max(distance, [], 'all');
spreedRate = (spreed - 1) / epoc;

for i = 1:epoc
    leraningRate = (LRMax - LRMin) * ((epoc - i) / (epoc - 1)) + LRMin;
    
    for j = 1:sizeOfDataSetX * trainP
        winner = zeros(1, SOMSizeRows * SOMSizeColumns);
        for k = 1:sizeOfCoordinatesY
            winner(k) = mtimes(dataSet.X(randP(j),:), squeeze(SOMNetwork(coordinates(1, k),coordinates(2, k),:)));
        end
        [~, winnerIndex] = max(winner);
        disp(winner);
        disp(winnerIndex);
        winnerDistance = distance(winnerIndex, :);
        distanceFunc = exp((-distance(winnerIndex).^2) / ((2 * spreed ^ 2)));

        SOMNetworkM = zeros(SOMSizeRows, SOMSizeColumns, sizeOfDataSetY);
        for k = 1:sizeOfCoordinatesY
            SOMNetworkM(coordinates(1, k), coordinates(2, k), :) = dataSet.X(randP(j),:)' - squeeze(SOMNetwork(coordinates(1, k),coordinates(2, k),:));
        end
    
        SOMNetwork = SOMNetwork + (leraningRate * distanceFunc * SOMNetworkM);
    end

    spreed = spreed - spreedRate;
end
