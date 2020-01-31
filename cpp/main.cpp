#include <Eigen/Dense>
#include <chrono>
#include "weight.hpp"

using namespace Eigen;
using namespace std;


int main() {
    string dir = "~/workspace/NutLearning/old_results/ \
        2020-01-29_20:02_hdim_64";
    Weight *w = new Weight(dir, 2, 64);

    MatrixXd mat(4000, 3);
    mat.setOnes();

    auto start = chrono::steady_clock::now();
    w->predict(mat);
    auto end = chrono::steady_clock::now();
    cout << "Elapsed time in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds> (end - start).count()
        << " ms" << endl;
}