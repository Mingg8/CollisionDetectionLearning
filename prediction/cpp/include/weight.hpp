#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "utils.hpp"

using namespace std;
using namespace Eigen;

class Weight {
  public:
    Weight(string dir, int lnum, int hdim);
    MatrixXd predict(MatrixXd pnts);

  private:
    int lnum;
    int hdim;
    string dir;
    vector<MatrixXd> weight;
    vector<VectorXd> bias;
    VectorXd input_coeff;
    VectorXd output_coeff;

    void loadWeight();
    void loadNormalizationCoeff();
    void normalize(MatrixXd &mat);
    void unnormalize(MatrixXd &mat);
    MatrixXd relu(MatrixXd);
    MatrixXd tanh_mat(MatrixXd);
    MatrixXd tanh_diff(MatrixXd mat);
    MatrixXd relu_diff(MatrixXd mat);
};