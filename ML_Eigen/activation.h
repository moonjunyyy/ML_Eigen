#pragma once
#include <vector>
#include <Eigen/Dense>

class activation
{
	int step(double);
	double sigmoid(double);
	std::vector<double> softmod(std::vector<double>);
	Eigen::VectorXd softmod(Eigen::VectorXd);
	static double* softmax(double* x, int n);
};

