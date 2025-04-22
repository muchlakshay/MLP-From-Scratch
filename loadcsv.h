//this module for loading csv files is inefficient, it first extract data into 2d std::vectors
//and then convert it to Eigen MatrixXd. While i can write it to directly write into Eigen MatrixXs
//but i am lazy!!

#pragma once

#include "loadcsv.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

struct Data {
    std::vector<std::string> header;
    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<double> Y_test;
    
};

//load data line by line from a csv file
Data load_csv(const std::string& filename, const std::string& target_column, double training_ratio){
    std::ifstream file (filename);
    std::string line;
    std::vector<std::string> header;

    Data data;

    int count {1};

    //runs for every line
    while(std::getline(file, line)){
        std::stringstream str_stream (line);
        std::string temp_str;

        //extract column names 
        if (count==1) {

            while (std::getline(str_stream,temp_str,',')){
                data.header.push_back(temp_str);
            }
            ++count; 
            continue;
        }

        std::vector<double> temp_vec;

        //extract data rows and store it in temp_vec
        while (std::getline(str_stream, temp_str, ',')){
            if (temp_str.empty()||temp_str==" "){
                temp_vec.push_back(0.0);
            }
            else {
                temp_vec.push_back(std::stod(temp_str));
            }
        }

        //data row is added in data vector
        (data.X_train).push_back(temp_vec);
    }
    file.close();

    //extract the index of target column i.e column to predict
    int target_index {};
    for (const auto& column : data.header){
        if (column==target_column){
            break;
        }
        ++target_index;
    }

    //copies target column in Y_train and deletes it from X_train
    for (auto& row : data.X_train){
        data.Y_train.push_back(row[target_index]);
        row.erase(row.begin()+target_index);
    }

    //Split data according to the ratio given as argument into X_tain, X_test, Y_train and Y_test
    //ik its inefficient
    int training_size {static_cast<int>(data.X_train.size()*training_ratio)};
    for (int i = data.X_train.size(); i>training_size; --i){
        data.X_test.push_back(data.X_train[i-1]);
        data.Y_test.push_back(data.Y_train[i-1]);
        data.X_train.erase(data.X_train.begin()+i);
        data.Y_train.erase(data.Y_train.begin()+i);
    }

    return data;
}

//convert 2d std::vector to Eigen MatrixXd;
Eigen::MatrixXd toMatrixXd(std::vector<std::vector<double>>& data){
    Eigen::MatrixXd mat (data.size(), data[0].size());
    for (std::size_t i {}; i<data.size(); ++i){
        mat.row(i) = Eigen::Map<Eigen::RowVectorXd>(data[i].data(), data[i].size());
    }
    return mat;
}

//convert std::vector to Eigen VectorXd;
Eigen::VectorXd toVectorXd(std::vector<double>& data){   
    return Eigen::Map<Eigen::VectorXd>(data.data(), data.size());
}

//one hot encode labels (for multiclass classification)
Eigen::MatrixXd toOneHot(Eigen::VectorXd& labels, int num_labels){
    Eigen::MatrixXd one_hot {Eigen::MatrixXd::Zero(labels.rows(), num_labels)};
    for (std::size_t i {}; i<labels.rows(); ++i){
        one_hot(i, static_cast<int>(labels(i))) = 1.0;
    }
    return one_hot;
}

//noramlize the data
void normalizeMatrixXd(Eigen::MatrixXd& matrix){
    matrix = (matrix.rowwise() - matrix.colwise().mean());
    matrix = matrix.array().rowwise() / matrix.colwise().norm().array();
}

