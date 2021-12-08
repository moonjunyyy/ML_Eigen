#include "activation.h"
#include <cmath>
int activation::step(double x) {
	if (x >= 0) {
		return 1;
	}
	else {
		return -1;
	}
}

double activation::sigmoid(double x) {
	return 1. / (1. + pow(2.71828, -x));
}

std::vector<double> activation::softmod(std::vector<double> x)
{
	std::vector<double> buf;
	buf.resize(x.size());
	double max = 0.;
	double sum = 0.;

	for (auto& d : x)
		if (max < d) max = d;

	for (int i = 0; i < x.size(); i++) {
		buf[i] = exp(x[i] - max);
		sum += buf[i];
	}
	for (int i = 0; i < x.size(); i++) {
		buf[i] /= sum;
	}

	return buf;
}

Eigen::VectorXd activation::softmod(Eigen::VectorXd x)
{
	Eigen::VectorXd buf;
	buf.resize(x.size());
	double max = 0.;
	double sum = 0.;

	for (auto& d : x)
		if (max < d) max = d;

	for (int i = 0; i < x.size(); i++) {
		buf[i] = exp(x[i] - max);
		sum += buf[i];
	}
	for (int i = 0; i < x.size(); i++) {
		buf[i] /= sum;
	}

	return buf;
}

double* activation::softmax(double* x, int n) {

	double* y = new double[n];
	double max = 0.;
	double sum = 0.;

	for (int i = 0; i < n; i++) {
		if (max < x[i]) {
			max = x[i];  // to prevent overflow
		}
	}

	for (int i = 0; i < n; i++) {
		y[i] = exp(x[i] - max);
		sum += y[i];
	}

	for (int i = 0; i < n; i++) {
		y[i] /= sum;
	}

	return y;
}