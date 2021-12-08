#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>

class EigenBinaryClassifyPerceptrons
{
public:

	int nIn;       // dimensions of input data
	Eigen::VectorXd w;  // weight vector of perceptrons

	EigenBinaryClassifyPerceptrons(int nIn);
	int train(Eigen::VectorXd x, int t, double learningRate);
	int predict(Eigen::VectorXd x);
private:
	int step(double);
};

class EigenLogisticRegression
{
public:
	int nIn;
	int nOut;
	int minibatchSize = 50;
	// private
	Eigen::MatrixXd grad_W; // = new double[nOut][nIn];
	Eigen::VectorXd grad_b; // = new double[nOut];
	Eigen::MatrixXd dY;		// = new double[minibatchSize][nOut];
	Eigen::MatrixXd W;		//
	Eigen::VectorXd b;		//

	EigenLogisticRegression(int n, int nO, int mini) {
		nIn = n;
		nOut = nO;
		minibatchSize = mini;
		int i, j;

		W.resize(nOut, nIn);
		b.resize(nOut);

		for (i = 0; i < nOut; i++) 
		{
			b[i] = 0.1;
			for (j = 0; j < nIn; j++) W(i, j) = 1.0; // initialize W, b
		}

		grad_W.resize(nOut, nIn);
		grad_b.resize(nOut);
		for (i = 0; i < nOut; i++) {
			grad_b[i] = 0.0;
			for (j = 0; j < nIn; j++) grad_W(i, j) = 0.0; // initialize grad_W, grad_b
		}

		dY.resize(minibatchSize, nOut);
	}
	EigenLogisticRegression(int n, int nO) {
		nIn = n;
		nOut = nO;
		int i, j;

		W.resize(nOut, nIn);
		b.resize(nOut);

		for (i = 0; i < nOut; i++)
		{
			b[i] = 0.1;
			for (j = 0; j < nIn; j++) W(i, j) = 1.0; // initialize W, b
		}

		grad_W.resize(nOut, nIn);
		grad_b.resize(nOut);
		for (i = 0; i < nOut; i++) {
			grad_b[i] = 0.0;
			for (j = 0; j < nIn; j++) grad_W(i, j) = 0.0; // initialize grad_W, grad_b
		}

		dY.resize(minibatchSize, nOut);
	}

	void train(double** X, int** T, int minibatchSize, double learningRate);
	double* output(double* x);
	int* predict(double* x);
};