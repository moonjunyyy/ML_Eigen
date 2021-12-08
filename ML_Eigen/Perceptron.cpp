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
