#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

int read_int(std::ifstream& stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (int(bytes[0]) << 24) | (int(bytes[1]) << 16) | (int(bytes[2]) << 8) | int(bytes[3]);
}

MatrixXd load_mnist_images(const std::string& filename, int& num_images, int& rows, int& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open image file!");

    int magic = read_int(file);
    num_images = read_int(file);
    rows = read_int(file);
    cols = read_int(file);

    MatrixXd images(num_images, rows * cols);

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images(i, j) = static_cast<double>(pixel) / 255.0; // normalize
        }
    }

    return images;
}

VectorXd load_mnist_labels(const std::string& filename, int& num_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open label file!");

    int magic = read_int(file);
    num_labels = read_int(file);

    VectorXd labels(num_labels);

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels(i) = static_cast<int>(label);
    }

    return labels;
}

//one hot encode labels
MatrixXd toOneHot(VectorXd& labels, int num_labels){
    MatrixXd one_hot {MatrixXd::Zero(labels.rows(), num_labels)};
    for (std::size_t i {}; i<labels.rows(); ++i){
        one_hot(i, static_cast<int>(labels(i))) = 1.0;
    }
    return one_hot;
}
