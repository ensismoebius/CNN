#include "NN.h"
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

neuralNework::NN::~NN()
{
}

//////////////////////////////////////////////////////
//////////////// Error functions /////////////////////
//////////////////////////////////////////////////////

double neuralNework::absError(double error)
{
    return std::abs(error);
}

double neuralNework::absErrorD(double error)
{
    return 1;
}

double neuralNework::simpleError(double error)
{
    return error;
}

double neuralNework::simpleErrorD(double error)
{
    return 1;
}

double neuralNework::quadraticError(double error)
{
    return std::pow(error, 2) / 2;
}

double neuralNework::quadraticErrorD(double error)
{
    return error;
}

neuralNework::NN::NN(ErrorFunction errorFunction)
{
    switch (errorFunction) {
    case AbsoluteError:
        this->errorFunction = absError;
        this->errorFunctionD = absErrorD;
        break;
    case SimpleError:
        this->errorFunction = simpleError;
        this->errorFunctionD = simpleErrorD;
        break;
    case QuadraticError:
        this->errorFunction = quadraticError;
        this->errorFunctionD = quadraticErrorD;
        break;
    }
}

bool neuralNework::NN::addLayer(unsigned nodes, LayerType type, ActivationFunction function)
{
    // A layer must have, at least, 1 node
    if (nodes < 1)
        return false;

    // The first layer must be an input
    if (this->layers.size() == 0 && type != LayerType::Input)
        return false;

    LayerProperties layer;
    layer.size = nodes;
    layer.type = type;

    switch (function) {
    case None:
        layer.activationFunction = none;
        layer.activationFunctionD = noneD;
        break;
    case Relu:
        layer.activationFunction = relu;
        layer.activationFunctionD = reluD;
        break;
    case Step:
        layer.activationFunction = step;
        layer.activationFunctionD = stepD;
        break;
    case Silu:
        layer.activationFunction = silu;
        layer.activationFunctionD = siluD;
        break;
    case Sigmoid:
        layer.activationFunction = sigmoid;
        layer.activationFunctionD = sigmoidD;
        break;
    case Hiperbolic:
        layer.activationFunction = hiperbolicTangent;
        layer.activationFunctionD = hiperbolicTangentD;
        break;
    case Softplus:
        layer.activationFunction = softplus;
        layer.activationFunctionD = softplusD;
        break;
    case LeakyRelu:
        layer.activationFunction = leakyRelu;
        layer.activationFunctionD = leakyReluD;
        break;
    }

    // Adding layers
    this->layers.push_back(layer);

    return true;
}

bool neuralNework::NN::assemble()
{
    this->layers[1].weights.resize(2, 2);
    this->layers[1].weights.at(0, 0) = 0.15;
    this->layers[1].weights.at(0, 1) = 0.20;
    this->layers[1].weights.at(1, 0) = 0.25;
    this->layers[1].weights.at(1, 1) = 0.30;

    this->layers[1].bias.resize(2, 1);
    this->layers[1].bias.at(0, 0) = 0.35;
    this->layers[1].bias.at(1, 0) = 0.35;

    this->layers[2].weights.resize(2, 2);
    this->layers[2].weights.at(0, 0) = 0.4;
    this->layers[2].weights.at(0, 1) = 0.45;
    this->layers[2].weights.at(1, 0) = 0.50;
    this->layers[2].weights.at(1, 1) = 0.55;

    this->layers[2].bias.resize(2, 1);
    this->layers[2].bias.at(0, 0) = 0.60;
    this->layers[2].bias.at(1, 0) = 0.60;

    //    ////////////////////////////////////////////////////
    //    ///// The weights creating and initializing ////////
    //    ////////////////////////////////////////////////////
    //    for (unsigned i = 1; i < this->layers.size(); i++) {
    //        this->layers[i].weights.resize(this->layers[i].size, this->layers[i - 1].size);
    //        this->layers[i].weights.randu();
    //        this->layers[i].bias.resize(this->layers[i].size, 1);
    //        this->layers[i].bias.randu();
    //    }
    return true;
}

void neuralNework::NN::showStructure(bool showMatrices)
{
    std::cout << "//////////////////////// Layers and weights ///////////////////////////" << std::endl;
    unsigned size = this->layers.size();

    // Each layer has its weights companions
    for (unsigned i = 0; i < size; i++) {
        std::cout << this->layers[i].size << "x" << 1 << " - Layer_" << i << std::endl;

        if (i == size - 1)
            break;

        std::cout << this->layers[i].weights.n_rows << "x" << this->layers[i].weights.n_cols << " - Weights_" << i << std::endl;
        if (showMatrices)
            std::cout << this->layers[i].weights << std::endl;
    }

    std::cout << "/////////////////////// Layers and weights end //////////////////////////" << std::endl;
}

arma::Mat<double> neuralNework::NN::feedForward(arma::Mat<double>& input)
{
    unsigned i = 0;

    // Apply activation function
    // output = activationFunction(zMatrix)
    arma::Mat<double> output = this->layers[i].weights * input; // Generate the zMatrix = (weights * input)
    applyActivationFunc(output, i + 1);

    unsigned size = this->layers.size();

    // Each layer has its weights companions except for the output layer
    for (i = 1; i < size; i++) {
        // Apply activation function
        // output = activationFunction(zMatrix)
        output = this->layers[i].weights * output; // Generate the zMatrix = (weights * input)
        applyActivationFunc(output, i + 1);
    }

    return output;
}

inline void neuralNework::NN::applyActivationFunc(arma::Mat<double>& value, unsigned index)
{
    value.transform(this->layers[index].activationFunction);
}

inline arma::Mat<double> neuralNework::NN::applyActivationFuncD(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->layers[index].activationFunctionD);
}

void neuralNework::NN::backPropagation(arma::Mat<double>& target, arma::Mat<double>& input)
{
    //
    arma::Mat<double> newWeights[this->layers.size()];

    // This will be important in the final part of the backpropagation
    this->layers[0].values = input;

    float learnningRate = 0.5;
    std::cout << "///////////// Learnning rate /////////////" << std::endl;
    std::cout << learnningRate << std::endl
              << std::endl;

    std::cout << "///////////// Target /////////////" << std::endl;
    std::cout << target << std::endl;

    ////////////////////
    /// Feed forward ///
    ////////////////////

    std::cout << "//////// Begin - Feed forward ///////" << std::endl;

    // Layer index
    int layerIndex = 1;

    std::cout << "Feed Forward: Layer 0 (Input) " << std::endl;
    std::cout << this->layers[0].values << std::endl;

    std::cout << "FeedForward: Weight " << layerIndex - 1 << " (InputToHiddenWeights) " << std::endl;
    std::cout << this->layers[layerIndex].weights << std::endl;

    this->layers[layerIndex].values = this->layers[layerIndex].weights * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
    this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);
    std::cout << "Feed Forward: Layer " << layerIndex << " (Hidden) " << std::endl;
    std::cout << this->layers[layerIndex].values << std::endl;

    layerIndex++;

    for (; layerIndex < this->layers.size(); layerIndex++) {

        std::cout << "FeedForward: Weight " << layerIndex - 1 << " (HiddenToOutputWeights) " << std::endl;
        std::cout << this->layers[layerIndex].weights << std::endl;

        this->layers[layerIndex].values = this->layers[layerIndex].weights * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
        this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);

        std::cout << "Feed Forward: Layer " << layerIndex << " (Hidden/Output) " << std::endl;
        std::cout << this->layers[layerIndex].values << std::endl;
    }

    std::cout << "//////// End - Feed forward ///////" << std::endl;

    ///////////////////////
    /// Backpropagation ///
    ///////////////////////
    layerIndex--;

    std::cout << "//////// begin - Back propagation ///////" << std::endl;

    // this is the neural network error, but in practice we
    // do not use the error itself, this is here just for
    // learnning purposes
    std::cout << "/n- Backprop: Error" << std::endl;
    arma::Mat<double> err = this->layers[layerIndex].values - target;
    std::cout << err.transform(this->errorFunction) << std::endl;

    // this is the derivative of the neural network error
    arma::Mat<double> error = this->layers[layerIndex].values - target;
    std::cout << "a- Backprop: Layer 2 errors (output errors)" << std::endl;
    std::cout << error << std::endl;

    ///////////////////////////////////////////////

    // this is the derivative of activation function with respect with its input (the z matrix)
    arma::Mat<double> activationSlope = applyActivationFuncD(this->layers[layerIndex].values, layerIndex);
    std::cout << "b- Backprop OUTPUT: Layer 2 activation function derivative" << std::endl;
    std::cout << activationSlope << std::endl;

    // this is the derivative of the Z matrix with respect with his weights (aka the value of the layer 1).
    std::cout << "c- Backprop OUTPUT: Layer 2 Zmatrix derivative (Layer 1 Transposed)" << std::endl;
    std::cout << this->layers[layerIndex - 1].values.t() << std::endl;

    std::cout << "d- Backprop OUTPUT: Layer 2 delta (a*b)" << std::endl;
    arma::Mat<double> delta = error % activationSlope;
    std::cout << delta << std::endl;

    std::cout << "e- Backprop OUTPUT: Layer 2 gradient (d*c)" << std::endl;
    // this is the derivative of the error with respect to the weights
    arma::Mat<double> gradient = delta * this->layers[layerIndex - 1].values.t();
    std::cout << gradient << std::endl;

    std::cout << "f- Backprop OUTPUT: New Weight 1 (hidden->output)" << std::endl;
    newWeights[layerIndex] = this->layers[layerIndex].weights - learnningRate * gradient;
    std::cout << this->layers[layerIndex].weights << "to" << std::endl
              << newWeights[layerIndex] << std::endl;

    layerIndex--;

    for (; layerIndex > 0; layerIndex--) {

        // this is the derivative of activation function with respect with its input (the z matrix)
        activationSlope = applyActivationFuncD(this->layers[layerIndex].values, layerIndex);
        std::cout << "b- Backprop HIDDEN: Layer " << layerIndex << " activation function derivative" << std::endl;
        std::cout << activationSlope << std::endl;

        // this is the derivative of the Z matrix with respect with his weights (aka the value of the layer 1).
        std::cout << "c- Backprop HIDDEN: Layer " << layerIndex << " Zmatrix derivative (Layer 1 Transposed)" << std::endl;
        std::cout << this->layers[layerIndex].values.t() << std::endl;

        std::cout << "d- Backprop HIDDEN: Layer " << layerIndex << " delta (a*b)" << std::endl;
        delta = this->layers[layerIndex + 1].weights.t() * delta % activationSlope;
        std::cout << delta << std::endl;

        // this is the derivative of the error with respect to the weights
        std::cout << "e- Backprop HIDDEN: Layer " << layerIndex << " gradient (d*c)" << std::endl;
        gradient = delta * this->layers[layerIndex - 1].values.t();
        std::cout << gradient << std::endl;

        std::cout << "f- Backprop HIDDEN: New Weight " << layerIndex << std::endl;
        newWeights[layerIndex] = this->layers[layerIndex].weights - learnningRate * gradient;
        std::cout << this->layers[layerIndex].weights << "to" << std::endl
                  << newWeights[layerIndex] << std::endl;
    }

    std::cout << "//////// end - Backpropagation ///////" << std::endl;

    for (layerIndex = 0; layerIndex < this->layers.size(); layerIndex++) {
        this->layers[layerIndex].weights = newWeights[layerIndex];
    }
}

////////////////////////////////////////////////////////////
//////////////// Activations functions /////////////////////
////////////////////////////////////////////////////////////

double neuralNework::sigmoid(double x)
{
    return 1.0 / (1.0 + std::pow(M_E, -x));
}
double neuralNework::sigmoidD(double x)
{
    return x * (1.0 - x);
}

double neuralNework::step(double x)
{
    return x < 0.0 ? 0.0 : 1.0;
}
double neuralNework::stepD(double x)
{
    return x != 0 ? 0.0 : std::numeric_limits<double>::quiet_NaN();
}

double neuralNework::relu(double x)
{
    //    std::cout << "relu" << std::endl;
    return std::max(x, 0.0);
}
double neuralNework::reluD(double x)
{
    //    std::cout << "reluD" << std::endl;
    return x < 0.0 ? 0.0 : 1.0;
}

double neuralNework::softplus(double x)
{
    return std::log(1.0 + std::pow(M_E, x));
}
double neuralNework::softplusD(double x)
{
    return 1.0 / (1.0 + std::pow(M_E, -x));
}

double neuralNework::leakyRelu(double x)
{
    return x < 0.0 ? 0.01 * x : x;
}
double neuralNework::leakyReluD(double x)
{
    return x < 0.0 ? 0.01 : 1.0;
}

double neuralNework::hiperbolicTangent(double x)
{
    std::cout << "hiperbolic" << std::endl;
    double y1 = std::pow(M_E, x);
    double y2 = std::pow(M_E, -x);
    return (y1 - y2) / (y1 + y2);
}
double neuralNework::hiperbolicTangentD(double x)
{
    std::cout << "hiperbolicD" << std::endl;
    return 1.0 - std::pow(hiperbolicTangent(x), 2.0);
}

double neuralNework::silu(double x)
{
    return 1.0 / (1.0 + std::pow(M_E, -x));
}
double neuralNework::siluD(double x)
{
    return (
               (1 + std::pow(M_E, -x)) + (x + std::pow(M_E, -x)))
        / (std::pow(1.0 + std::pow(M_E, -x), 2));
}

double neuralNework::none(double x)
{
    std::cout << "None" << std::endl;
    return x;
}
double neuralNework::noneD(double x)
{
    std::cout << "NoneD" << std::endl;
    return 0;
}
