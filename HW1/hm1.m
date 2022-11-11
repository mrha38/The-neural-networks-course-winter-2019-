'Implementation of "f(x) = sin(x) + x.cos(3x)" function using MATLAB toolbox'

x=[-6:0.01:6]
t=sin(x)+x*cos(3*x)'

net = newff(x,t,[25])
net.layers{1}.transferFcn = 'logsig'
net.divideFcn = 'dividerand' 
net.divideParam.trainRatio = 70/100
net.divideParam.valRatio = 10/100
net.divideParam.testRatio = 20/100
net.trainParam.lr = 0.01;
net.trainParam.show=25;
net.trainParam.epochs=1000;
net.trainParam.max_fail=50;
net=init(net);
net = train(net,x,t)
