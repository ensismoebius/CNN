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

double neuralNework::simpleError(double error)
{
    return error;
}

double neuralNework::quadraticError(double error)
{
    return std::pow(error, 2) / 2;
}

neuralNework::NN::NN(ErrorFunction errorFunction)
{
    switch (errorFunction) {
    case AbsoluteError:
        this->errorFunction = absError;
        break;
    case SimpleError:
        this->errorFunction = simpleError;
        break;
    case QuadraticError:
        this->errorFunction = quadraticError;
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
    //////////////////////////////////////////////////////
    /////// The weights creating and initializing ////////
    //////////////////////////////////////////////////////
    for (unsigned i = 0; i < this->layers.size() - 1; i++) {
        this->layers[i].weights.resize(this->layers[i + 1].size, this->layers[i].size);
        this->layers[i].weights.randu();
    }
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
    auto output = applyActivationFunc(
        this->layers[i].weights * input, // Generate the zMatrix = (weights * input)
        i + 1 // Index of activation function
    );

    unsigned size = this->layers.size();

    // Each layer has its weights companions except for the output layer
    for (i = 1; i < size; i++) {
        // Apply activation function
        // output = activationFunction(zMatrix)
        output = applyActivationFunc(
            this->layers[i].weights * output, // Generate the zMatrix = (weights * input)
            i + 1 // Index of activation function
        );
    }

    return output;
}

inline arma::Mat<double> neuralNework::NN::applyActivationFunc(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->layers[index].activationFunction);
}

inline arma::Mat<double> neuralNework::NN::applyActivationFuncD(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->layers[index].activationFunctionD);
}

void neuralNework::NN::backPropagation(arma::Mat<double>& target, arma::Mat<double>& input)
{

    float learnningRate = 0.01;

    ////////////////////
    /// Feed forward ///
    ////////////////////

    // Preparing to store the intermediate values
    const unsigned size = this->layers.size();
    std::vector<arma::Mat<double>> layers(size);

    unsigned layerIndex = 0;
    // Apply activation function
    // output = activationFunction(zMatrix)
    layers[layerIndex] = applyActivationFunc(
        this->layers[layerIndex].weights * input, // Generate the zMatrix = (weights * input)
        layerIndex + 1 // Index of activation function
    );

    // Each layer has its weights except for the output layer
    for (layerIndex = 1; layerIndex < size; layerIndex++) {
        // Apply activation function
        // output = activationFunction(zMatrix)
        layers[layerIndex] = applyActivationFunc(
            this->layers[layerIndex].weights * layers[layerIndex - 1], // Generate the zMatrix = (weights * input)
            layerIndex + 1 // Index of activation function
        );
    }

    ///////////////////////
    /// Backpropagation ///
    ///////////////////////

    layerIndex--; // Points to output
    arma::Mat<double> errors = target - layers[layerIndex];

    // delta = gradient * hidden.tranposed
    arma::Mat<double> gradientHiddenToOutput = (this->applyActivationFuncD(layers[layerIndex - 1], layerIndex) % errors) * learnningRate;
    arma::Mat<double> deltaHiddenToOutput = gradientHiddenToOutput * layers[layerIndex].t();

    // Update the weights
    this->layers[layerIndex].weights += deltaHiddenToOutput;
    // Update bias weights
    //    this->biasOutput += gradientHiddenToOutput;

    for (; layerIndex > 0; layerIndex++) {
        layers[layerIndex].transform(this->layers[layerIndex].activationFunctionD);
        this->layers[layerIndex].weights += (layers[layerIndex] % errors) * learnningRate * layers[layerIndex].t();
        errors = this->layers[layerIndex].weights.t() + layers[layerIndex];
    }
}

////////////////////////////////////////////////////////////
//////////////// Activations functions /////////////////////
////////////////////////////////////////////////////////////

double neuralNework::sigmoid(double x)
{
    std::cout << "sigmoid" << std::endl;
    return 1.0 / (1.0 + std::pow(M_E, -x));
}
double neuralNework::sigmoidD(double x)
{
    std::cout << "sigmoidD" << std::endl;
    double y = neuralNework::sigmoid(x);
    return y * (1.0 - y);
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
    std::cout << "relu" << std::endl;
    return std::max(x, 0.0);
}
double neuralNework::reluD(double x)
{
    std::cout << "reluD" << std::endl;
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
