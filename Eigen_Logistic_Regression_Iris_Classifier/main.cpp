#include <vector>
#include <random>
#include <fstream>
#include <chrono>

#pragma comment (lib,"../ML_Eigen/x64/Release/ML_Eigen.lib")
#include "../ML_Eigen/Perceptron.h"

using namespace std;

int train_N;  // number of training data
int test_N;   // number of test data
const double testRatio = 0.2;
const int nIn  = 4;        // dimensions of input data
const int nOut = 3;
int max_Epochs = 2000;
const int minibatchSize = 30;
double learningRate = 0.1;

int main(int argc, char* argv[])
{
	/*Build Data*/

	vector<Eigen::VectorXd> train_X;
	vector<Eigen::VectorXi>	train_Y;
	vector<Eigen::VectorXd> test_X;
	vector<Eigen::VectorXi>	test_Y;
	vector<Eigen::VectorXi> predict_Y;

	fstream fio("iris_data.txt", ios::in);
	while (true)
	{
		double f1, f2, f3, f4;
		int c;
		fio >> f1 >> f2 >> f3 >> f4 >> c;

		if (fio.eof()) break;
		Eigen::VectorXd buf;
		buf.resize(nIn);
		buf << f1, f2, f3, f4;
		train_X.push_back(buf);

		Eigen::VectorXi cls;
		cls.resize(nOut);
		cls << (c == 0), (c == 1), (c == 2);
		train_Y.push_back(cls);
	}
	fio.close();

	train_N = train_X.size();

	test_N = train_N * testRatio;
	train_N -= test_N;

	cout << train_X.size() << endl;

	/*Divide Random Test Data*/
	default_random_engine generator;
	uniform_real_distribution<> udist(0., 1.);

	for (int i = 0; i < test_N; i++)
	{
		int index = udist(generator) * train_X.size();
		test_X.push_back(train_X[index]);
		train_X.erase(std::next(train_X.begin(), index));
		test_Y.push_back(train_Y[index]);
		train_Y.erase(std::next(train_Y.begin(), index));
	}
	
	cout << "Train Data (" << train_X.size() << ") : " << endl;
	for (auto& V : train_X)
	{
		for (auto& D : V)
		{
			cout << D << '\t';
		}
		cout << endl;
	}
	cout << endl;

	cout << "Test Data (" << test_X.size() << ") : " << endl;
	for (auto& V : test_X)
	{
		for (auto& D : V)
		{
			cout << D << '\t';
		}
		cout << endl;
	}
	cout << endl;

	/*Train model*/
	EigenLogisticRegression ELR(nIn, nOut, minibatchSize);
	
	int epoch = 0;

	while (true)
	{
		for (int i = 0; i < train_N; i += minibatchSize)
		{
			ELR.train(train_X.begin() + i, train_Y.begin() + i, minibatchSize, learningRate);
		}
		if (epoch == max_Epochs) break;
		epoch++;
		learningRate *= 0.95;
	}

	cout << "After " << epoch << " / " << max_Epochs << " iterations, Weight is : \n";
	cout << ELR.W << endl << endl;

	/*Test model*/
	for (auto& V : test_X)
	{
		predict_Y.push_back(ELR.predict(V));
	}

	int confusionMatrix[3][4][4];
	for (int i = 0; i < 3; i++)
		confusionMatrix[i][0][0] = confusionMatrix[i][0][1] = confusionMatrix[i][1][0] = confusionMatrix[i][1][1] = 0;

	for (int i = 0; i < test_X.size(); i++)
		for (int c = 0; c < 3; c++)
			confusionMatrix[c][(test_Y[i](1) + test_Y[i](2) * 2) == c][(predict_Y[i](1) + predict_Y[i](2) * 2) == c] += 1;

	std::cout << "------------------------------------" << endl;
	std::cout << "Logistic Regression model evaluation" << endl;
	std::cout << "------------------------------------" << endl;

	for (int c = 0; c < 3; c++)
	{
		double TP = confusionMatrix[c][1][1];
		double TN = confusionMatrix[c][0][0];
		double FP = confusionMatrix[c][0][1];
		double FN = confusionMatrix[c][1][0];

		cout << "for Class" << c << endl;
		std::cout << "Accuracy:\t" << 100. * (TP + TN) / test_N << endl;
		std::cout << "Precision:\t" << 100. * TP / (TP + FP) << endl;
		std::cout << "Recall:\t" << 100. * TP / (TP + FN) << endl << endl;
	}

	std::cout << "------------------------------------" << endl;
	std::cout << "Estimate Running Time Between Method" << endl;
	std::cout << "------------------------------------" << endl;

	/*Copy Data into C Style Array*/
	double** cTrain_X, ** cTest_X;
	int** cTrain_Y, ** cTest_Y, **cPredict_Y;

	cTrain_X	= new double* [train_N];
	cTrain_Y	= new int* [train_N];
	for (int i = 0; i < train_N; i++) { cTrain_X[i] = new double[nIn]; cTrain_Y[i] = new int[nOut]; }
	cTest_X		= new double* [test_N];
	cTest_Y		= new int* [test_N];
	cPredict_Y	= new int* [test_N];
	for (int i = 0; i < test_N; i++) { cTest_X[i] = new double[nIn]; cTest_Y[i] = new int[nOut]; }

	for (int i = 0; i < train_N; i++)
	{
		for (int j = 0; j < nIn; j++)
			cTrain_X[i][j] = train_X[i][j];
		for (int j = 0; j < nOut; j++)
			cTrain_Y[i][j] = train_Y[i][j];
	}
	for (int i = 0; i < test_N; i++)
	{
		for (int j = 0; j < nIn; j++)
			cTest_X[i][j] = test_X[i][j];
		for (int j = 0; j < nOut; j++)
			cTest_Y[i][j] = test_Y[i][j];
	}

	/*Build Model Using Cstyle Array & Eigen Vector Data*/
	max_Epochs = 2000;
	learningRate = 0.2;

	LogisticRegression		LRClass(nIn, nOut, minibatchSize);
	EigenLogisticRegression	ELRClass(nIn, nOut, minibatchSize);

	epoch = 0;

	chrono::system_clock::time_point start = std::chrono::system_clock::now();
	while (true)
	{
		for (int i = 0; i < train_N; i += minibatchSize)
		{
			LRClass.train(cTrain_X + i, cTrain_Y + i, minibatchSize, learningRate);
		}
		if (epoch == max_Epochs) break;
		epoch++;
		learningRate *= 0.95;
	}
	std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;

	cout << "for Traditional Method, " << epoch << " / " << max_Epochs << " iterations take " << sec.count() << " sec.\n";
	
	for (int i = 0; i < test_N; i++)
		cPredict_Y[i] = LRClass.predict(cTest_X[i]);
	
	for (int i = 0; i < 3; i++)
		confusionMatrix[i][0][0] = confusionMatrix[i][0][1] = confusionMatrix[i][1][0] = confusionMatrix[i][1][1] = 0;

	for (int i = 0; i < test_X.size(); i++)
		for (int c = 0; c < 3; c++)
			confusionMatrix[c][(cTest_Y[i][1] + cTest_Y[i][2] * 2) == c][(cPredict_Y[i][1] + cPredict_Y[i][2] * 2) == c] += 1;

	std::cout << "------------------------------------" << endl;
	std::cout << "Logistic Regression model evaluation" << endl;
	std::cout << "------------------------------------" << endl;

	for (int c = 0; c < 3; c++)
	{
		double TP = confusionMatrix[c][1][1];
		double TN = confusionMatrix[c][0][0];
		double FP = confusionMatrix[c][0][1];
		double FN = confusionMatrix[c][1][0];

		cout << "for Class" << c << endl;
		std::cout << "Accuracy:\t" << 100. * (TP + TN) / test_N << endl;
		std::cout << "Precision:\t" << 100. * TP / (TP + FP) << endl;
		std::cout << "Recall:\t" << 100. * TP / (TP + FN) << endl << endl;
	}

	for (int i = 0; i < train_N; i++) { delete [] cTrain_X[i]; delete[] cTrain_Y[i]; }
	for (int i = 0; i < test_N; i++) { delete[] cTest_X[i]; delete[] cTest_Y[i]; delete[] cPredict_Y[i]; }
	delete[] cTrain_X, delete[] cTest_X, delete[] cTrain_Y, delete[] cTest_Y, delete[] cPredict_Y;

	epoch = 0;
	learningRate = 0.2;

	start = std::chrono::system_clock::now();
	while (true)
	{
		for (int i = 0; i < train_N; i += minibatchSize)
		{
			ELRClass.train(train_X.begin() + i, train_Y.begin() + i, minibatchSize, learningRate);
		}
		if (epoch == max_Epochs) break;
		epoch++;
		learningRate *= 0.95;
	}
	sec = std::chrono::system_clock::now() - start;
	cout << "for Eigen Method, " << epoch << " / " << max_Epochs << " iterations take " << sec.count() << " sec.\n";

	for (int i = 0; i < 3; i++)
		confusionMatrix[i][0][0] = confusionMatrix[i][0][1] = confusionMatrix[i][1][0] = confusionMatrix[i][1][1] = 0;

	for (int i = 0; i < test_X.size(); i++)
		for (int c = 0; c < 3; c++)
			confusionMatrix[c][(test_Y[i](1) + test_Y[i](2) * 2) == c][(predict_Y[i](1) + predict_Y[i](2) * 2) == c] += 1;

	std::cout << "------------------------------------" << endl;
	std::cout << "Logistic Regression model evaluation" << endl;
	std::cout << "------------------------------------" << endl;

	for (int c = 0; c < 3; c++)
	{
		double TP = confusionMatrix[c][1][1];
		double TN = confusionMatrix[c][0][0];
		double FP = confusionMatrix[c][0][1];
		double FN = confusionMatrix[c][1][0];

		cout << "for Class" << c << endl;
		std::cout << "Accuracy:\t" << 100. * (TP + TN) / test_N << endl;
		std::cout << "Precision:\t" << 100. * TP / (TP + FP) << endl;
		std::cout << "Recall:\t" << 100. * TP / (TP + FN) << endl << endl;
	}

	return 0;
}