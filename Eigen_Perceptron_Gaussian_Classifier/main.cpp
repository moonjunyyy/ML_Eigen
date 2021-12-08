#include <vector>
#include <random>
#include <fstream>

#pragma comment (lib,"../ML_Eigen/x64/Release/ML_Eigen.lib")
#include "../ML_Eigen/Perceptron.h"

using namespace std;

const int train_N	= 1000;  // number of training data
const int test_N	= 200;   // number of test data
const int nIn		= 2;        // dimensions of input data
const double learningRate = 1.;

int main(int argc, char* argv[])
{
	/*Build Data*/

	vector<Eigen::VectorXd> train_X;
	vector<int>				train_Y;
	vector<Eigen::VectorXd> test_X;
	vector<int>				test_Y;
	vector<int>				predict_Y;

	train_X.resize(train_N); train_Y.resize(train_N);
	test_X.resize(test_N); test_Y.resize(test_N); predict_Y.resize(test_N);

	default_random_engine generator;
	normal_distribution<double> dist1(-2., 1.5);
	normal_distribution<double> dist2( 2., 1.5);

	ofstream of("Gauss_Data.txt");
	for (int i = 0; i < train_N / 2; i++)
	{
		train_X[i].resize(2);
		train_X[i](0) = dist1(generator);
		train_X[i](1) = dist2(generator);
		train_Y[i] = 1;

		of << train_X[i](0) << ',' << train_X[i](1) << endl;
	}
	of << endl;
	for (int i = train_N / 2; i < train_N; i++)
	{
		train_X[i].resize(2);
		train_X[i](0) = dist2(generator);
		train_X[i](1) = dist1(generator);
		train_Y[i] = -1;

		of << train_X[i](0) << ',' << train_X[i](1) << endl;
	}
	of << endl;
	for (int i = 0; i < test_N / 2; i++)
	{
		test_X[i].resize(2);
		test_X[i](0) = dist1(generator);
		test_X[i](1) = dist2(generator);
		test_Y[i] = 1;

		of << test_X[i](0) << ',' << test_X[i](1) << endl;
	}
	of << endl;
	for (int i = test_N / 2; i < test_N; i++)
	{
		test_X[i].resize(2);
		test_X[i](0) = dist2(generator);
		test_X[i](1) = dist1(generator);
		test_Y[i] = -1;

		of << test_X[i](0) << ',' << test_X[i](1) << endl;
	}
	of.close();

	/*Train Model*/

	EigenBinaryClassifyPerceptrons classifier(nIn);

	const int max_Epoch = 2000;
	int epoch = 0;

	while (epoch < max_Epoch)
	{
		int classified = 0;
		
		for (int i = 0; i < train_N; i++)
		{
			classified += classifier.train(train_X[i], train_Y[i], learningRate);
		}
		if (classified == train_N) break;
		epoch++;
	}
	cout << "Epoch : " << epoch << " / " << max_Epoch << endl;
	cout << "Wieghts :\n" << classifier.w << endl;

	/*Predict Data*/

	for (int i = 0; i < test_N; i++)
		predict_Y[i] = classifier.predict(test_X[i]);

	/*Evaluate Data*/
	int confusionMatrix[2][2];
	double accuracy = 0.;
	double precision = 0.;
	double recall = 0.;
	confusionMatrix[0][0] = confusionMatrix[0][1] = confusionMatrix[1][0] = confusionMatrix[1][1] = 0;

	for (int i = 0; i < test_N; i++)
		confusionMatrix[test_Y[i] > 0][predict_Y[i] > 0] += 1;

	int TP = confusionMatrix[1][1];
	int TN = confusionMatrix[0][0];
	int FP = confusionMatrix[0][1];
	int FN = confusionMatrix[1][0];

	std::cout << "----------------------------" << endl;
	std::cout << "Perceptrons model evaluation" << endl;
	std::cout << "----------------------------" << endl;

	std::cout << "Accuracy:\t" << 100. * (TP + TN) / (double)test_N << endl;
	std::cout << "Precision:\t" << 100. * TP / (double)(TP + FP) << endl;
	std::cout << "Recall:\t" << 100. * TP / (double)(TP + FN) << endl;

	return 0;
}