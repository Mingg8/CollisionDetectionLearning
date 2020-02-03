#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

MatrixXd readCsv(string file, int rows, int cols);
VectorXd readCsv_vec(std::string file, int rows);
