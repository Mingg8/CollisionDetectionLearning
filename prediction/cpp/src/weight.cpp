#include "weight.hpp"

Weight::Weight(string _dir, int _lnum, int _hdim)
: lnum(_lnum), hdim(_hdim), dir(_dir) {
    loadWeight();
    loadNormalizationCoeff();
};

MatrixXd Weight::predict(MatrixXd pnts) {
    normalize(pnts);
    MatrixXd a1, a2, a3, a4, res;
    MatrixXd normal;
    
    // cout << "size: " << weight[0].cols() << " " << weight[1].rows() << endl;
    // cout << weight[0] << endl;

    a1 = (pnts * weight[0]).rowwise() + bias[0].transpose(); // N x 64
    a2 = (relu(a1) * weight[1]).rowwise() + bias[1].transpose(); // N x 64
    a3 = (relu(a2) * weight[2]).rowwise() + bias[2].transpose(); // N x 64
    a4 = (relu(a3) * weight[3]).rowwise() + bias[3].transpose(); // N x 1
    res = tanh_mat(a4);

    MatrixXd dh4 = weight[3] * tanh_diff(a4).transpose();
    MatrixXd dh3 = weight[2] * (relu_diff(a3).transpose()).cwiseProduct(dh4);
    MatrixXd dh2 = weight[1] * (relu_diff(a2).transpose()).cwiseProduct(dh3);
    MatrixXd dh1 = weight[0] * (relu_diff(a1).transpose()).cwiseProduct(dh2);

    unnormalize(res);
    unnormalize(dh1);

    return dh1;
}

void Weight::loadWeight() {
    int num = 0;

    string file = dir + string("/weight") + std::to_string(num) + string(".csv");
    num++;
    MatrixXd W = readCsv(file, 3, 64);

    file = dir + string("/weight") + to_string(num) + string(".csv");
    num++;
    VectorXd b = readCsv_vec(file, hdim);

    weight.push_back(W);
    bias.push_back(b);
    for (int i = 0; i < lnum; i++) {
        file = dir + string("/weight") + to_string(num) + string(".csv");
        num++;
        W = readCsv(file, hdim, hdim);

        file = dir + string("/weight") + to_string(num) + string(".csv");
        num++;
        b = readCsv_vec(file, hdim);

        weight.push_back(W);
        bias.push_back(b);
    }

    file = dir + string("/weight") + to_string(num) + string(".csv");
    num++;
    W = readCsv(file, hdim, 4);
    MatrixXd W_reduced(hdim, 1);
    W_reduced = W.col(0);

    file = dir + string("/weight") + to_string(num) + string(".csv");
    num++;
    b = readCsv_vec(file, 4);
    VectorXd b_reduced(1);
    b_reduced(0) = b(0);

    weight.push_back(W_reduced);
    bias.push_back(b_reduced);
}

void Weight::loadNormalizationCoeff() {
    string file_input = dir + string("/input_coeff.csv");
    string file_output = dir + string("/output_coeff.csv");
    input_coeff = readCsv_vec(file_input, 6);
    output_coeff = readCsv_vec(file_output, 8);
}

void Weight::normalize(MatrixXd &mat) {
    int n = mat.rows();
    Vector3d a = input_coeff.head(3);
    Vector3d b = input_coeff.tail(3);
    for (int i = 0; i < 3; i++) {
        mat.col(i) = mat.col(i) * a(i) + b(i) * VectorXd::Ones(n);
    }
}

void Weight::unnormalize(MatrixXd &mat) {
    int n = mat.rows();
    Vector3d a = output_coeff.head(4);
    Vector3d b = output_coeff.tail(4);
    mat = (mat - b(0) * VectorXd::Ones(n)) / a(0);
}

MatrixXd Weight::relu(MatrixXd mat) {
    MatrixXd res = MatrixXd::Zero(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); i ++) {
        for (int j = 0; j < mat.cols(); j++) {
            if (mat(i, j) >= 0) {
                res(i, j) = mat(i, j);
            } else {
                res(i, j) = 0;
            }
        }
    }
    return res;
}

MatrixXd Weight::tanh_mat(MatrixXd mat) {
    MatrixXd res = MatrixXd::Zero(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            res(i, j) = tanh(mat(i,j));
            cout << mat(i,j ) << " " << res(i,j)  << endl;
        }
    }
    return res;
}

MatrixXd Weight::tanh_diff(MatrixXd mat) {
    MatrixXd res = MatrixXd::Zero(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            double a = (exp(mat(i, j) + exp(-mat(i, j)))) / 2;
            res(i, j) = 1 / a;
        }
    }
    return res;
}

MatrixXd Weight::relu_diff(MatrixXd mat) {
    MatrixXd res = MatrixXd::Zero(mat.rows(), mat.cols());
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if (mat(i, j) >= 0) {
                res(i, j) = 1;
            }
        }
    }
    return res;
}