#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "activation.h"

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
	std::vector<Eigen::VectorXd> dY;		// = new double[minibatchSize][nOut];
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

		dY.resize(minibatchSize);
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

		dY.resize(minibatchSize);
	}

	void train(std::vector<Eigen::VectorXd>::iterator x, std::vector<Eigen::VectorXi>::iterator T, int minibatchSize, double learningRate);
	Eigen::VectorXd output(Eigen::VectorXd x);
	Eigen::VectorXi predict(Eigen::VectorXd x);
};


class LogisticRegression {
public:
	int nIn;
	int nOut;
	int minibatchSize = 50;
	// private
	double** grad_W; // = new double[nOut][nIn];
	double* grad_b; // = new double[nOut];
	double** dY; // = new double[minibatchSize][nOut];
	double** W;
	double* b;

	LogisticRegression(int n, int nO, int mini) {
		;
		nIn = n;
		nOut = nO;
		minibatchSize = mini;
		int i, j;

		W = new double* [nOut];
		b = new double[nOut];
		for (i = 0; i < nOut; i++) {
			W[i] = new double[nIn];
			b[i] = 0.1;
			for (j = 0; j < nIn; j++) W[i][j] = 1.0; // initialize W, b
		}

		grad_W = new double* [nOut];
		grad_b = new double[nOut];
		for (i = 0; i < nOut; i++) {
			grad_W[i] = new double[nIn];
			grad_b[i] = 0.0;
			for (j = 0; j < nIn; j++) grad_W[i][j] = 0.0; // initialize grad_W, grad_b
		}

		dY = new double* [minibatchSize];  for (int i = 0; i < minibatchSize; i++) dY[i] = new double[nOut];
		//LogisticRegression(n, nO);
	}
	LogisticRegression(int n, int nO) {
		nIn = n;
		nOut = nO;
		int i, j;

		W = new double* [nOut];
		for (i = 0; i < nOut; i++) {
			W[i] = new double[nIn];
			for (j = 0; j < nIn; j++) W[i][j] = 1.0; // initialize W
		}
		b = new double[nOut]; for (i = 0; i < nOut; i++) b[i] = 0.1; // initialize b

		grad_W = new double* [nOut];  for (int i = 0; i < nOut; i++) grad_W[i] = new double[nIn];
		grad_b = new double[nOut];

		dY = new double* [minibatchSize];  for (int i = 0; i < minibatchSize; i++) dY[i] = new double[nOut];
		/*for (int i = 0; i < nOut; i++) {
			for (int j = 0; j < nIn; j++) cout << W[i][j] << "  ";
			cout << " W  b = " << b[i] << endl;
		}*/

	}
	void train(double** X, int** T, int minibatchSize, double learningRate);
	double* output(double* x);
	int* predict(double* x);
};