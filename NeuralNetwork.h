#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

using namespace Eigen;

class NeuralNetwork{
    struct ActivationFunction{
        std::function<double(double)> fn;
        std::function<double(double)> derivative;
    };
    
    struct LossFunction{
        std::function<VectorXd(MatrixXd, MatrixXd)> loss;
        std::function<MatrixXd(MatrixXd, MatrixXd)> derivative;
    };

    std::vector<MatrixXd> weights;
    std::vector<VectorXd> bias;
    std::vector<MatrixXd> weights_gradients;
    std::vector<VectorXd> bias_gradients;
    std::vector<MatrixXd> layers_output;
    std::vector<MatrixXd> layers_activations;
    std::vector<MatrixXd> layers_errors;

    double learning_rate {};

    const std::vector<int> layers;
    std::vector<std::string> activation_function;

    void initialize(const std::vector<int>& layers);
    ActivationFunction getFunction(int layer);
    LossFunction getLossFunction(const std::string& loss);
    double cross_entropy_loss(const MatrixXd& y, const MatrixXd& y_hat);
    MatrixXd softmax(const MatrixXd& z);

    void shuffle(const MatrixXd& non_shuffled_features, MatrixXd& store_shuffled_features, 
                 const MatrixXd& non_shuffled_labels, MatrixXd& store_shuffled_labels);

    void feedForward(const MatrixXd& batch);
    void backpropagate(const MatrixXd& batch, const MatrixXd& labels, int batch_size, const std::string& loss_function);
    void updateWeights();

    double computeLoss(const MatrixXd& outputs, const MatrixXd& labels, int batch_size, const std::string& loss_function);

public:

    NeuralNetwork(const std::vector<int>& layers, const std::vector<std::string>& activation_function)
    :layers{layers}, activation_function {activation_function}{
        initialize(layers);
    }

    void learn(const MatrixXd& training_data, const MatrixXd& training_labels, int epochs,
        double eta, int batch_size, const std::string& loss_function, bool verbose=true);

    MatrixXd predict(const MatrixXd& to_predict);
    const std::vector<MatrixXd>& getWeights() const {return weights;}
};

double accuracy(const MatrixXd& labels, const MatrixXd& predicted);



