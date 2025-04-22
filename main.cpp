#include "NeuralNetwork.h"
#include "load_mnist.h"
#include "loadcsv.h"

int main(){

//--------------LOAD CSV FILES------------------------

    // Data data {load_csv("filename.csv", "label_column", 0.8)}; //--> change training data ratio from 
                                                                   // 0.8 to something else if you want

    // MatrixXd X_train {toMatrixXd(data.X_train)};
    // MatrixXd X_test {toMatrixXd(data.X_test)};
    // VectorXd Y_train {toVectorXd(data.Y_train)};
    // VectorXd Y_test {toVectorXd(data.Y_test)};

//you can use toOneHot() to one hot encode the labels

//--------------LOAD MNIST----------------------------
    
    int num_images_train, rows_train, cols_train, num_labels_train;
    int num_images_test, rows_test, cols_test, num_labels_test;

    //change path to where your MNIST data is 
    MatrixXd train_images = load_mnist_images("MNIST/train-images.idx3-ubyte", num_images_train, rows_train, cols_train);
    MatrixXd test_images = load_mnist_images("MNIST/t10k-images.idx3-ubyte", num_images_test, rows_test, cols_test);
    VectorXd train_labels = load_mnist_labels("MNIST/train-labels.idx1-ubyte", num_labels_train);
    VectorXd test_labels = load_mnist_labels("MNIST/t10k-labels.idx1-ubyte", num_labels_test);

    MatrixXd train_labels_oh {toOneHot(train_labels, 10)};
    MatrixXd test_labels_oh {toOneHot(test_labels, 10)};

    NeuralNetwork nn {{784, 16, 16, 10}, {"relu", "relu", "softmax"}};
    nn.learn(train_images, train_labels_oh, 30, 0.01, 32, "cross_entropy_loss", true);

    MatrixXd pred {nn.predict(test_images)};

    for (int i {}; i<test_labels_oh.rows(); ++i){
        std::cout<<"Actual: "<<test_labels_oh.row(i)<<" Predicted: ";
        for (int j {}; j<10; ++j){
            std::cout<<std::round(pred(i,j))<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<"Accuracy: "<<accuracy(test_labels_oh, pred);

    return 0;
}
