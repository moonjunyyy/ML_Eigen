import numpy as np

class pyLogisticRegression:

    def __init__(self, nin, nout, mini):
        
        self.nIn = nin;
        self.nOut = nout;
        self.miniBatchSize = mini;

        self.W = np.matrix(np.zeros((self.nOut, self.nIn)));
        for d in self.W :
            d = 1;

        self.b = np.matrix(np.zeros(self.nOut));
        for d in self.b :
            d = 0.1;

        self.grad_W = np.matrix(np.zeros((self.nOut, self.nIn)));
        for d in self.grad_W :
            d = 0.;

        self.grad_b = np.matrix(np.zeros(self.nOut));
        for d in self.grad_b :
            d = 0.;

        self.dY = np.matrix(np.zeros((self.miniBatchSize,self.nOut)));

    def train(self, X, Y, batchSize, learningRate):
        for n in range(0, batchSize):
            predicted_Y_ = self.output(X[n])
            self.dY[n] = predicted_Y_ - Y[n]
            self.grad_W += np.transpose(np.matrix(self.dY[n])) * np.matrix(X[n]);
            self.grad_b += self.dY[n]

        self.W -= self.grad_W * learningRate / batchSize;
        self.b -= self.grad_b * learningRate / batchSize;

    def output(self, X):
        R = self.softMax(self.W * np.transpose(np.matrix(X)) + np.transpose(self.b));
        return R;

    def predict(self, X):
        Buf = self.output(X);
        max = 0.;
        index = 0;

        for i in range(Buf.size()):
            if max < Buf[i]:
                max = Buf[i];
                index = i;

        for i in range(Buf.size()):
            if i == index :
                Buf[i] = 1;
            else :
                Buf[i] = 0;

        return Buf;

    def softMax(self, X):
        y = np.zeros(X.size);
        max = 0.;
        sum = 0.;
        
        for d in X:
            if max < d:
                max = d;

        for i in range(0,X.size):
            y[i] = np.exp(X[i] - max);
            sum += y[i];

        y /= sum;
        return y;
