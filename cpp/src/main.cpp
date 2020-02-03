#include <Eigen/Dense>
#include <chrono>
#include "weight.hpp"

using namespace Eigen;
using namespace std;


int main() {
    string dir = "/home/mjlee/workspace/NutLearning/old_results/2020-01-29_20:02_hdim_64/weight_csv";
    Weight *w = new Weight(dir, 2, 64);

    // MatrixXd mat(4000, 3);
    // mat.setOnes();

    MatrixXd mat(2, 3); 
    mat << 0.0165, 0.0165, 0.0371, 0, 0, 0;

    auto start = chrono::steady_clock::now();
    MatrixXd output = w->predict(mat);
    
    auto end = chrono::steady_clock::now();
    cout << "Elapsed time in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds> (end - start).count()
        << " ms" << endl;

    cout << output << endl;
}