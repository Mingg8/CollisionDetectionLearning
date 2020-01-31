#include "weight.hpp"

Weight::Weight(string _dir, int _lnum, int _hdim)
: lnum(_lnum), hdim(_hdim), dir(_dir) {
    loadWeight();
};

MatrixXd Weight::predict(MatrixXd pnts) {
    normalize(pnts);
    MatrixXd res(pnts.rows(), 1);

    res = relu((pnts * weight[0]).rowwise() + bias[0].transpose());
    res = relu((res * weight[1]).rowwise() + bias[1].transpose());
    res = relu((res * weight[2]).rowwise() + bias[2].transpose());
    res = tanh_mat((res * weight[3]).rowwise() + bias[3].transpose());
    unnormalize(res);

    return res;
}

MatrixXd Weight::readCsv(std::string file, int rows, int cols) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;

    MatrixXd res = MatrixXd(rows, cols);
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            int len = line.length();

            cols = 0;

            char *start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == ',') {
                    res(row, col++) = atof(start);
                    start = ptr + 1 + i;
                }
            }
            res(row, col) = atof(start);
            row++;
        }
        in.close();
    }
    return res;
}

VectorXd Weight::readCsv_vec(std::string file, int rows) {
    ifstream in(file);
    string line;

    int row = 0;

    VectorXd res(rows);
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            res(row) = atof(ptr);
            row++;
        }
        in.close();
    }
    return res;
}

void Weight::loadWeight() {
    int num_file = (2 + lnum) * 2;
    int num = 0;

    string file = dir + string("/weight") + to_string(num) + string(".csv");
    num++;
    MatrixXd W = readCsv(file, 3, hdim);

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

    file = dir + string("/weight") + to_string(num) + string(".csv");
    num++;
    b = readCsv_vec(file, 4);

    weight.push_back(W);
    bias.push_back(b);
}

void Weight::loadNormalizationCoeff() {
    string file_input = dir + string("/input_coeff.csv");
    string file_output = dir + string("/output_coeff.csv");
    input_coeff = readCsv_vec(file_input, 6);
    output_coeff = readCsv_vec(file_output, 8);
}

void Weight::normalize(MatrixXd &mat) {
}

void Weight::unnormalize(MatrixXd &mat) {

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
        }
    }
    return res;
}