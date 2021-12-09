#include "Perceptron.h"

EigenBinaryClassifyPerceptrons::EigenBinaryClassifyPerceptrons(int n)
{
	nIn = n;
	w.resize(nIn);
	for (auto& d : w) d = .1;
}

int EigenBinaryClassifyPerceptrons::train(Eigen::VectorXd x, int t, double learningRate)
{
	if (x.size() != w.size()) throw;
	double c = w.dot(x) * t;
	
	if (c > 0) return 1;

	w += learningRate * t * x;
	return 0;
}

int EigenBinaryClassifyPerceptrons::predict(Eigen::VectorXd x)
{
	if (x.size() != w.size()) throw;
	return step(w.dot(x));
}

int EigenBinaryClassifyPerceptrons::step(double d)
{
	if (d > 0) return 1;
	return -1;
}

void EigenLogisticRegression::train(std::vector<Eigen::VectorXd>::iterator X, std::vector<Eigen::VectorXi>::iterator T, int minibatchSize, double learningRate)
{
	for (int n = 0; n < minibatchSize; n++) {

		Eigen::VectorXd predicted_Y_ = output(X[n]);
		dY[n] = predicted_Y_ - T[n].cast<double>();
		grad_W += dY[n] * X[n].transpose();
		grad_b += dY[n];
	}

	// 2. update params
	W -= learningRate * grad_W / minibatchSize;
	b -= learningRate * grad_b / minibatchSize;
}

Eigen::VectorXd EigenLogisticRegression::output(Eigen::VectorXd x)
{
	activation myAct;
	Eigen::VectorXd R;
	R = myAct.softmax(W * x + b);
	return R;
}

Eigen::VectorXi EigenLogisticRegression::predict(Eigen::VectorXd x)
{
	Eigen::VectorXd Buf;
	Buf = output(x);
	double max = 0.;
	int index = 0;

	Eigen::VectorXi R;
	R.resize(Buf.size());

	for (int i = 0; i < x.size(); i++)
		if (Buf(i) > max) { max = Buf(i); index = i; }

	for (int i = 0; i < x.size(); i++)
	{
		if (i == index) { R(i) = 1; }
		else { R[i] = 0; }
	}
	return R;
}

void LogisticRegression::train(double** X, int** T, int minibatchSize, double learningRate) {
	// train with SGD
	// 1. calculate gradient of W, b
	for (int n = 0; n < minibatchSize; n++) {

		double* predicted_Y_ = output(X[n]);

		for (int j = 0; j < nOut; j++) {
			dY[n][j] = predicted_Y_[j] - T[n][j];

			for (int i = 0; i < nIn; i++) {
				grad_W[j][i] += dY[n][j] * X[n][i];
			}

			grad_b[j] += dY[n][j];
		}
		delete predicted_Y_;// Java --> C++
	}

	// 2. update params
	for (int j = 0; j < nOut; j++) {
		for (int i = 0; i < nIn; i++) {
			W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
		}
		b[j] -= learningRate * grad_b[j] / minibatchSize;
	}
	//return dY;
}

double* LogisticRegression::output(double* x) {
	activation myAct;
	double* preActivation = new double[nOut];
	for (int i = 0; i < nOut; i++) preActivation[i] = 0;

	for (int j = 0; j < nOut; j++) {

		for (int i = 0; i < nIn; i++) {
			preActivation[j] += W[j][i] * x[i];
		}

		preActivation[j] += b[j];  // linear output
	}

	return myAct.softmax(preActivation, nOut);
}

int* LogisticRegression::predict(double* x) {

	double* y = output(x);  // activate input data through learned networks
	int* t = new int[nOut]; // output is the probability, so cast it to label

	int argmax = -1;
	double max = 0.;

	for (int i = 0; i < nOut; i++) {
		if (max < y[i]) {
			max = y[i];
			argmax = i;
		}
	}

	for (int i = 0; i < nOut; i++) {
		if (i == argmax) {
			t[i] = 1;
		}
		else {
			t[i] = 0;
		}
	}
	return t;
}
