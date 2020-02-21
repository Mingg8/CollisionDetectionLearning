#include "utils.hpp"

MatrixXd readCsv(std::string file, int rows, int cols) {
    ifstream in(file);
    string line;

    int row = 0;
    int col = 0;
    
    MatrixXd res = MatrixXd(rows, cols);
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *)line.c_str();
            int len = line.length();

            col = 0;

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

VectorXd readCsv_vec(std::string file, int rows) {
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