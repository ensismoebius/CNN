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

bool neuralNework::NN::addLayer(unsigned nodes, LayerType type, ActivationFunction activation)
{
    // A layer must have, at least, 1 node
    if (nodes < 1)
        return false;

    // The first layer must be an input
    if (this->layers.size() == 0 && type != LayerType::Input)
        return false;

    LayerProperties layer;
    layer.activation = activation;
    layer.size = nodes;
    layer.type = type;

    switch (activation) {
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
    //    this->layers[1].weights.resize(2, 2);
    //    this->layers[1].weights.at(0, 0) = 0.15;
    //    this->layers[1].weights.at(0, 1) = 0.20;
    //    this->layers[1].weights.at(1, 0) = 0.25;
    //    this->layers[1].weights.at(1, 1) = 0.30;

    //    this->layers[1].bias.resize(2, 1);
    //    this->layers[1].bias.at(0, 0) = 0.35;
    //    this->layers[1].bias.at(1, 0) = 0.35;

    //    this->layers[2].weights.resize(2, 2);
    //    this->layers[2].weights.at(0, 0) = 0.4;
    //    this->layers[2].weights.at(0, 1) = 0.45;
    //    this->layers[2].weights.at(1, 0) = 0.50;
    //    this->layers[2].weights.at(1, 1) = 0.55;

    //    this->layers[2].bias.resize(2, 1);
    //    this->layers[2].bias.at(0, 0) = 0.60;
    //    this->layers[2].bias.at(1, 0) = 0.60;

    ////////////////////////////////////////////////////
    ///// The weights creating and initializing ////////
    ////////////////////////////////////////////////////
    for (unsigned i = 1; i < this->layers.size(); i++) {
        this->layers[i].weightsToPrevious.resize(this->layers[i].size, this->layers[i - 1].size);
        this->layers[i].weightsToPrevious.randu();
        this->layers[i].bias.resize(this->layers[i].size, 1);
        this->layers[i].bias.randu();
    }
    return true;
}

void neuralNework::NN::showStructure(bool showValues)
{
    std::cout << "//////////////////////// Layers and weights ///////////////////////////" << std::endl;
    unsigned size = this->layers.size();

    for (unsigned i = 0; i < size; i++) {

        // Input layers has no weights
        if (this->layers[i].type != neuralNework::LayerType::Input) {
            std::cout << this->layers[i].weightsToPrevious.n_rows << "x" << this->layers[i].weightsToPrevious.n_cols << " - Weights from layer" << i << " to layer" << i - 1 << std::endl;
            if (showValues)
                std::cout << this->layers[i].weightsToPrevious << std::endl;
        }

        // Layers data
        std::cout << this->layers[i].size << "x" << 1 << " - Layer " << i << std::endl;
        if (showValues)
            std::cout << this->layers[i].values << std::endl;
    }

    std::cout << "/////////////////////// Layers and weights end //////////////////////////" << std::endl;
}

arma::Mat<double> neuralNework::NN::feedForward(arma::Mat<double>& input)
{
    int layerIndex = 0;

    // First layer is the input
    this->layers[layerIndex].values = input;

    layerIndex = 1;

    this->layers[layerIndex].values = this->layers[layerIndex].weightsToPrevious * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
    this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);

    // Feed forward on next ones
    for (layerIndex = 2; layerIndex < this->layers.size(); layerIndex++) {
        this->layers[layerIndex].values = this->layers[layerIndex].weightsToPrevious * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
        this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);
    }

    layerIndex--;

    return this->layers[layerIndex].values;
}

inline void neuralNework::NN::applyActivationFunc(arma::Mat<double>& value, unsigned index)
{
    value.transform(this->layers[index].activationFunction);
}

inline arma::Mat<double> neuralNework::NN::applyActivationFuncD(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->layers[index].activationFunctionD);
}

void neuralNework::NN::backPropagation(arma::Mat<double>& target, arma::Mat<double>& input, float learnningRate)
{

    // Layer index
    int layerIndex = 0;

    // First layer is the input
    this->layers[layerIndex].values = input;

    // Stores the new weigths
    arma::Mat<double> newWeights[this->layers.size()];

    ////////////////////
    /// Feed forward ///
    ////////////////////

    // Feed forward on second layer: The layer zero is not modifiable so, we skip it
    layerIndex = 1;
    this->layers[layerIndex].values = this->layers[layerIndex].weightsToPrevious * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
    this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);

    // Feed forward on next ones
    for (layerIndex = 2; layerIndex < this->layers.size(); layerIndex++) {
        this->layers[layerIndex].values = this->layers[layerIndex].weightsToPrevious * this->layers[layerIndex - 1].values + this->layers[layerIndex].bias;
        this->applyActivationFunc(this->layers[layerIndex].values, layerIndex);
    }

    ///////////////////////
    /// Backpropagation ///
    ///////////////////////

    // Backpropagation on last layer

    // Pointing to last layer
    layerIndex--;

    // this is the derivative of the neural network error, in practice we do not use the error itself
    arma::Mat<double> error = this->layers[layerIndex].values - target;
    //    std::cout << "\rError: " << arma::accu(arma::abs(error)) << "\n Result: \n"
    //              << this->layers[layerIndex].values << std::flush;

    // this is the derivative of activation function with respect with its input (the z matrix)
    arma::Mat<double> activationSlope = applyActivationFuncD(this->layers[layerIndex].values, layerIndex);

    arma::Mat<double> delta = error % activationSlope;

    // this is the derivative of the error with respect to the weights
    arma::Mat<double> gradient = delta * this->layers[layerIndex - 1].values.t();

    //    std::cout << "f- Backprop OUTPUT: New Weight 1 (hidden->output)" << std::endl;
    newWeights[layerIndex] = this->layers[layerIndex].weightsToPrevious - learnningRate * gradient;
    //    std::cout << newWeights[layerIndex] << std::endl;

    // Backpropagation on remainning ones
    layerIndex--;
    for (; layerIndex > 0; layerIndex--) {

        // this is the derivative of activation function with respect with its input (the z matrix)
        activationSlope = applyActivationFuncD(this->layers[layerIndex].values, layerIndex);

        // delta
        delta = this->layers[layerIndex + 1].weightsToPrevious.t() * delta % activationSlope;

        // this is the derivative of the error with respect to the weights
        gradient = delta * this->layers[layerIndex - 1].values.t();

        //        std::cout << "f- Backprop HIDDEN: New Weight " << layerIndex << std::endl;
        newWeights[layerIndex] = this->layers[layerIndex].weightsToPrevious - learnningRate * gradient;
        //        std::cout << newWeights[layerIndex] << std::endl;
    }

    for (layerIndex = 0; layerIndex < this->layers.size(); layerIndex++) {
        this->layers[layerIndex].weightsToPrevious = newWeights[layerIndex];
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
    return std::max(x, 0.0);
}
double neuralNework::reluD(double x)
{
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
    double y1 = std::pow(M_E, x);
    double y2 = std::pow(M_E, -x);
    return (y1 - y2) / (y1 + y2);
}
double neuralNework::hiperbolicTangentD(double x)
{
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
    return x;
}
double neuralNework::noneD(double x)
{
    return 0;
}
