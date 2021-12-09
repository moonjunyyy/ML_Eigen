#pragma once
#include <vector>
#include <Eigen/Dense>

class activation
{
public:
	int step(double);
	double sigmoid(double);
	std::vector<double> softmax(std::vector<double>);
	Eigen::VectorXd softmax(Eigen::VectorXd);
	static double* softmax(double* x, int n);
};

