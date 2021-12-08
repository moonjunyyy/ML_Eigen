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

void EigenLogisticRegression::train(std::vector<Eigen::VectorXd> X, std::vector<Eigen::VectorXi> T, int minibatchSize, double learningRate)
{
	for (int n = 0; n < minibatchSize; n++) {

		Eigen::VectorXd predicted_Y_;
		output(X[n], predicted_Y_);

		dY[n] = predicted_Y_ - T[n];
		grad_W += dY[n] * X[n];
		grad_b += dY[n];
	}

	// 2. update params
	W -= learningRate / minibatchSize * grad_W;
	b -= learningRate / minibatchSize * grad_b;
}

void EigenLogisticRegression::output(Eigen::VectorXd x, Eigen::VectorXd& y)
{
	activation myAct;
	Eigen::VectorXd R;
	myAct.softmax(W * x + b, y);
}

void EigenLogisticRegression::predict(Eigen::VectorXd x, Eigen::VectorXi& y)
{
	Eigen::VectorXd R;
	output(x, R);
	double max = 0.;
	int index = 0;

	for (int i = 0; i < R.size(); i++)
		if (R(i) > max) { max = R(i); index = i; }

	for (int i = 0; i < R.size(); i++)
	{
		if (i == index) { y[i] = 1; }
		else { y[i] = 0; }
	}
}
