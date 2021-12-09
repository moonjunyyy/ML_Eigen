import numpy as np
import time
import LogisticRegression as LR

rawData = np.loadtxt("iris_data.txt");
np.random.shuffle(rawData);
rawData = np.transpose(rawData);
X = np.transpose(rawData[0:4])
rawY = np.transpose(rawData[4])

x = rawY.shape;
Y = np.empty((x[0], 3), int)


testV = np.matrix([[0,1,2],[3,4,5]]);
testW = np.matrix([3,4,5]);

testM = np.matrix(testV) * np.transpose(np.matrix(testW));
print(testM);



for i in range(0, x[0]):
    Y[i,0] = int(rawY[i] == 0)
    Y[i,1] = int(rawY[i] == 1)
    Y[i,2] = int(rawY[i] == 2)

trainX = X[0:120] 
trainY = Y[0:120]
testX  = X[120:150] 
testY  = Y[120:150]

classifier = LR.pyLogisticRegression(4,3,30);
Max_Epoch = 2000;
epoch = 0;
learningRate = 0.2;

start = time.time();

while epoch <= Max_Epoch:
    index = 0;
    while index < 120:
        classifier.train(trainX[index:index+30],trainY[index:index+30], 30, learningRate);
        index += 30;
    epoch += 1;
    learningRate *= 0.95;
    print(epoch, " / ", Max_Epoch);

end = time.time() - start;
print(end)

print(classifier.W)
