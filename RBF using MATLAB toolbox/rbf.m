clc
clear all
dataSet = importdata('iris.mat');
values = normalize(dataSet.X');

target = dataSet.class';
v = size(values);
t = size(target);
net = newrb(values, target, 0.01,1, 100);
y = sim(net, values);
view(net);