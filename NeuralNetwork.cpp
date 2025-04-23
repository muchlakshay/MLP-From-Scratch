#include "NeuralNetwork.h"

//----------------------MODEL INITIALIZATION----------------------

void NeuralNetwork::initialize(const std::vector<int>& layers){

    std::random_device rd;
    std::mt19937 mt(rd());

    for (std::size_t l {1}; l < layers.size(); ++l){
        int in = layers[l-1]; //layer input size
        int out = layers[l]; //layer output size

        //xavier xavier initialization
        double limit = std::sqrt(6.0 / (in + out));
        std::uniform_real_distribution dist(-limit, limit); //xavier limit

        //Create a matrix of weights according to the layer input and output size
        MatrixXd w(out, in);
        for (int row {}; row < w.rows(); ++row){
            for (int col {}; col < w.cols(); ++col){
                w(row, col) = dist(mt);
            }
        }

        //Push weight matrix of layer into a vector that contains weights of all layers
        weights.push_back(w);

        //Zero initialize the bias vector according to layer output size
        //and push back into vector containing bias of all layers 
        bias.push_back(VectorXd::Zero(out));

        //dummy initialize gradients for easy access according to layer index
        weights_gradients.push_back(MatrixXd::Zero(out, in));
        bias_gradients.push_back(VectorXd::Zero(out));

        //dummy initialize layer error for easy access according to layer index
        layers_errors.push_back(MatrixXd::Zero(1,out));
    }
}

//----------------------GET ACTIVATION FUNCTIONS----------------------

NeuralNetwork::ActivationFunction NeuralNetwork::getFunction(int layer){

    //activation function lambda with respective derivatives

    auto sigmoid {[](double z){return 1.0/(1.0+std::exp(-z));}};
    auto sigmoid_prime {[](double z){return (1.0/(1.0+std::exp(-z)))*(1.0-(1.0/(1.0+std::exp(-z))));}};

    auto tanh {[](double z){return std::tanh(z);}};
    auto tanh_prime{[](double z){return 1-std::tanh(z)*std::tanh(z);}};

    auto relu {[](double z){return std::max(0.0, z);}};
    auto relu_prime {[](double z){return z>0?1.0:0.0;}};

    auto leaky_relu {[](double z){return z>0?z:z*0.01;}};
    auto leaky_relu_prime {[](double z){return z>0?1.0:0.01;}};

    auto linear {[](double z){return z;}};
    auto linear_prime{[](double z){return 1.0;}};

    //activation function of the layer 
    std::string fn {activation_function[layer]};

    //return appropriate activation function and its derivative
    if (fn=="sigmoid") return {sigmoid, sigmoid_prime};
    else if (fn=="tanh") return {tanh, tanh_prime};
    else if (fn=="relu") return {relu, relu_prime};
    else if (fn=="leaky_relu") return {leaky_relu, leaky_relu_prime};
    else if (fn=="linear") return {linear, linear_prime};
    else return {};
}

//----------------------GET LOSS FUNCTION----------------------

NeuralNetwork::LossFunction NeuralNetwork::getLossFunction(const std::string& loss){

    //loss functions lambdas and their derivatives

    auto mse = [](const MatrixXd y, const MatrixXd y_hat) -> VectorXd {
        VectorXd losses(y.rows());
        for (int i = 0; i < y.rows(); ++i) {
            losses(i) = 0.5 * (y.row(i) - y_hat.row(i)).squaredNorm(); // squaredNorm = sum of squares
        }
        return losses;
    };
    auto mse_prime {
        [] (MatrixXd y, MatrixXd y_hat) -> MatrixXd {
            return y_hat-y;
        }
    };

    auto binary_cross_entropy{
        [](MatrixXd y, MatrixXd y_hat) -> VectorXd {
            for (int row {}; row<y_hat.rows(); ++row){
                for (int col {}; col<y_hat.cols(); ++col){
                    y_hat(row, col) = std::clamp(y_hat(row, col), 1e-7, 1 - 1e-7);
                }
            }
            VectorXd losses(y.rows());
            for (int i = 0; i < y.rows(); ++i) {
                losses(i) = - (y.row(i).array() * y_hat.row(i).array().log() + 
                                (1 - y.row(i).array()) * (1 - y_hat.row(i).array()).log()).sum();
            }
            return losses;
        }
    };
    auto binary_cross_entropy_prime{
        [](MatrixXd y, MatrixXd y_hat) -> MatrixXd {
            for (int row {}; row<y_hat.rows(); ++row){
                for (int col {}; col<y_hat.cols(); ++col){
                    y_hat(row, col) = std::clamp(y_hat(row, col), 1e-7, 1 - 1e-7);
                }
            }
            return (y_hat - y).array() / (y_hat.array() * (1 - y_hat.array()));
        }
    };

    //return appropriate loss function and its derivative
    if (loss=="mse") return {mse, mse_prime};
    else if (loss=="binary_cross_entropy") return {binary_cross_entropy, binary_cross_entropy_prime};
    else return {};
}

//----------------------SEPARATE FUNCTION FOR CROSS ENTROPY----------------------

double NeuralNetwork::cross_entropy_loss(const MatrixXd& y, const MatrixXd& y_hat) {
    const double epsilon = 1e-15;
    MatrixXd y_hat_clamped = y_hat.array().min(1.0 - epsilon).max(epsilon);
    double loss = 0.0;
    for (int i = 0; i < y.rows(); ++i) {
        loss -= (y.row(i).array() * y_hat_clamped.row(i).array().log()).sum();
    }
    return loss;
}

//----------------------SEPARATE FUNCTION FOR SOFTMAX----------------------
MatrixXd NeuralNetwork::softmax(const MatrixXd& z){
    MatrixXd result = z;
    for (int i = 0; i < z.rows(); ++i) {
        double max = z.row(i).maxCoeff();
        VectorXd exp_row = (z.row(i).array() - max).exp();
        double sum = exp_row.sum();
        result.row(i) = exp_row / sum;
    }
    return result;
}

//----------------------SHUFFLE DATASET----------------------

void NeuralNetwork::shuffle(const MatrixXd& non_shuffled_features, MatrixXd& store_shuffled_features, 
    const MatrixXd& non_shuffled_labels, MatrixXd& store_shuffled_labels){

    std::random_device rd;
    std::mt19937 mt(rd());

    //Indicies of Data  
    std::vector<int> indicies(non_shuffled_features.rows());
    for (std::size_t i {}; i<non_shuffled_features.rows(); ++i){
        indicies[i] = i;
    }

    //Shuffle the indicies
    std::shuffle(indicies.begin(), indicies.end(), mt);

    //using shuffled indicies store data it another matrix (both data features and labels)  
    for (std::size_t i {}; i<indicies.size(); ++i){
        store_shuffled_features.row(i)=non_shuffled_features.row(indicies[i]);
        store_shuffled_labels.row(i)=non_shuffled_labels.row(indicies[i]);
    }
}

//---------------------FEED FORWARD----------------------

void NeuralNetwork::feedForward(const MatrixXd& batch){

    //Check if batch is of suitable dimensions
    if (batch.cols() != layers[0]){
        std::cout<<"Input data size doesn't match the input layer size.\n";
        return;
    }

    //clear the previous batch outputs (z) and activations (a)
    layers_output.clear();
    layers_activations.clear();

    //input to layer (initially the batch for first layer)
    MatrixXd input_to_layer {batch};

    //Runs for every layer weight matrix 
    for (std::size_t l {}; l<weights.size(); ++l){

        //calculate the weighted sum
        MatrixXd weighted_sum {input_to_layer*weights[l].transpose()};
        weighted_sum.rowwise() += bias[l].transpose();

        //get suitable activation function according to layer index
        auto func {getFunction(l)};

        //layer activations
        MatrixXd activations;

        //for output layer if it has softmax activation 
        if (activation_function[l]=="softmax"){
            activations = softmax(weighted_sum);
        }
        //for hidden layers
        else{
            auto func {getFunction(l)};
            activations = weighted_sum.unaryExpr(func.fn);
        }

        //Push back the weighted sum (z) into the std::vector contating outputs of all layers
        layers_output.push_back(weighted_sum);

        //Push back the activations (a) into the std::vector contating activations of all layers
        layers_activations.push_back(activations);

        //Set input for next layer to output of current layer
        input_to_layer = layers_activations[l];
    }

}

//----------------------BACKPROPAGATION----------------------

void NeuralNetwork::backpropagate(const MatrixXd& batch, const MatrixXd& labels, int batch_size, const std::string& loss_function){

    //Starts from output layer (-2 because layer data member contains input layer too)
    for (int l {static_cast<int>(layers.size())-2}; l>=0; --l){

        //Get appropriate loss and activation function for their derivatives
        auto func {getFunction(l)};
        auto loss {getLossFunction(loss_function)};

        //if layer is output layer, calculate error of output layer
        if (l==layers.size()-2){

            //number of rows in output layer
            int rows {static_cast<int>(layers_activations.back().rows())};

            // ∂C/∂a
            MatrixXd loss_wrt_activations (batch_size, labels.cols());
            // ∂a/∂z
            MatrixXd activations_wrt_output;

            //if activation is softmax and loss is cross entropy
            if (activation_function.back()=="softmax" && loss_function=="cross_entropy_loss"){
                loss_wrt_activations = layers_activations.back()-labels;

                //store error in vector containg errors of all layers
                layers_errors[l] = loss_wrt_activations;
            }
            //if activation is sigmoid and loss is binary cross entropy
            else if (activation_function.back()=="sigmoid" && loss_function=="binary_cross_entropy"){
                loss_wrt_activations = layers_activations.back()-labels;
                layers_errors[l] = loss_wrt_activations;
            }
            //for any other loss and activation
            else {
                loss_wrt_activations = loss.derivative(labels, layers_activations.back());
                activations_wrt_output = layers_output.back().unaryExpr(func.derivative);
                //store error in vector containg errors of all layers
                layers_errors[l] = loss_wrt_activations.cwiseProduct(activations_wrt_output);
            }
            continue;
        }

        //calculate errors of hidden layers and store it in vector containg errors of all layers
        layers_errors[l] = (layers_errors[l+1]*weights[l+1]).cwiseProduct(layers_output[l].unaryExpr(func.derivative));
    }

    //runs for each layer's weight matrix
    for (int l {}; l<weights.size(); ++l){

        //if output layer
        if (l==0){
            //compute gradients
            weights_gradients[l] = layers_errors[l].transpose()*batch;
            //avegrage them 
            weights_gradients[l]/=batch_size;
        }
        //if hidden layer
        else {
            //compute gradients
            weights_gradients[l] = layers_errors[l].transpose()*layers_activations[l-1];
            //average them
            weights_gradients[l]/=batch_size;
        }
        //computer bias gradients;
        bias_gradients[l] = layers_errors[l].colwise().mean();
    }
} 

//----------------------UPDATE WEIGHTS----------------------

void NeuralNetwork::updateWeights(){
    for (std::size_t i {}; i<weights.size(); ++i){
        weights[i] -= learning_rate*weights_gradients[i];
        bias[i] -= learning_rate*bias_gradients[i];
    }
}

//----------------------LOSS COMPUTATION----------------------

double NeuralNetwork::computeLoss(const MatrixXd& outputs, const MatrixXd& labels, int batch_size, const std::string& loss_function){
    double loss {};
    if (loss_function=="cross_entropy_loss"){
        loss = cross_entropy_loss(labels, outputs);
        loss/=batch_size;
        return loss;
    }
    auto loss_func {getLossFunction(loss_function)};
    for (std::size_t i {}; i<batch_size; ++i){
        loss += loss_func.loss(labels, outputs).sum();
    }
    loss/=batch_size;
    return loss;
}

//----------------------TRAINING FUNCTION----------------------

void NeuralNetwork::learn(const MatrixXd& training_data, const MatrixXd& training_labels, int epochs,
                         double eta, int batch_size, const std::string& loss_function, bool verbose){
    
    //Separate Matricies to store shuffled features and labels 
    MatrixXd training_data_s (training_data.rows(), training_data.cols());
    MatrixXd training_labels_s (training_labels.rows(), training_labels.cols());

    learning_rate=eta;

    std::cout<<"Learning..\n";

    int rows {static_cast<int>(training_data_s.rows())};

    //Runs for each epoch
    for (int e {1}; e<=epochs; ++e){

        //shuffle features and labels
        shuffle(training_data, training_data_s, training_labels, training_labels_s);
        double loss {};
        
        //Divide whole dataset into batches
        for (int start {}; start<rows; start+=batch_size){

            //current_batch_size sets to number remaining rows if rows left are less then batch_size 
            int current_batch_size = std::min(batch_size, rows-start);

            //Create Batch of current_batch_size
            MatrixXd batch_features {training_data_s.block(start, 0, current_batch_size, training_data_s.cols())};
            MatrixXd batch_labels {training_labels_s.block(start, 0, current_batch_size, training_labels_s.cols())};

            //Feed forward the batch
            feedForward(batch_features);

            //Backpropagate the Errors and Calculate Gradients
            backpropagate(batch_features, batch_labels, current_batch_size, loss_function);

            //update Weights using gradients;
            updateWeights();

            //compute the loss
            loss+=computeLoss(layers_activations.back(), batch_labels, current_batch_size, loss_function);
        }
        if (verbose && e%5==0){
            std::cout<<"Epoch: "<<e<<" Loss: "<<loss<<"\n";
        }
    
    }
    std::cout << "Learning complete!\n";
}

//----------------------PREDICTION FUNCTION----------------------

MatrixXd NeuralNetwork::predict(const MatrixXd& to_predict){
        feedForward(to_predict);
        return layers_activations.back();
}

//----------------------ACCURACY CALCULATION----------------------
double accuracy(const MatrixXd& labels, const MatrixXd& predicted){
    int correct = 0;
    for (int i = 0; i < labels.rows(); ++i){
        Eigen::Index label_index, pred_index;
        labels.row(i).maxCoeff(&label_index);
        predicted.row(i).maxCoeff(&pred_index);
        if (label_index == pred_index) ++correct;
    }
    return correct / static_cast<double>(labels.rows());
}
