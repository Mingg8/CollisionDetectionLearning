#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>

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

    MatrixXd readCsv(std::string file, int rows, int cols);
    VectorXd readCsv_vec(std::string file, int rows);
    void loadWeight();
    void loadNormalizationCoeff();
    void normalize(MatrixXd &mat);
    void unnormalize(MatrixXd &mat);
    MatrixXd relu(MatrixXd);
    MatrixXd tanh_mat(MatrixXd);
};