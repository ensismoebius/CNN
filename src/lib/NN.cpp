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
    if (this->layersTypes.size() == 0 && type != LayerType::Input)
        return false;

    // Adding layers
    this->layersTypes.push_back(type);
    this->layersSizes.push_back(nodes);

    switch (function) {
    case None:
        this->activationFunctions.push_back(none);
        this->activationFunctionsD.push_back(noneD);
        break;
    case Relu:
        this->activationFunctions.push_back(relu);
        this->activationFunctionsD.push_back(reluD);
        break;
    case Step:
        this->activationFunctions.push_back(step);
        this->activationFunctionsD.push_back(stepD);
        break;
    case Silu:
        this->activationFunctions.push_back(silu);
        this->activationFunctionsD.push_back(siluD);
        break;
    case Sigmoid:
        this->activationFunctions.push_back(sigmoid);
        this->activationFunctionsD.push_back(sigmoidD);
        break;
    case Hiperbolic:
        this->activationFunctions.push_back(hiperbolicTangent);
        this->activationFunctionsD.push_back(hiperbolicTangentD);
        break;
    case Softplus:
        this->activationFunctions.push_back(softplus);
        this->activationFunctionsD.push_back(softplusD);
        break;
    case LeakyRelu:
        this->activationFunctions.push_back(leakyRelu);
        this->activationFunctionsD.push_back(leakyReluD);
        break;
    }

    return true;
}

bool neuralNework::NN::assemble()
{
    //////////////////////////////////////////////////////
    /////// The weights creating and initializing ////////
    //////////////////////////////////////////////////////
    for (unsigned i = 0; i < this->layersSizes.size() - 1; i++) {
        arma::Mat<double> hiddenWeight(this->layersSizes[i + 1], this->layersSizes[i]);
        hiddenWeight.randu();

        this->networkWeights.push_back(hiddenWeight);
    }
    return true;
}

void neuralNework::NN::showStructure(bool showMatrices)
{
    std::cout << "//////////////////////// Layers and weights ///////////////////////////" << std::endl;
    unsigned size = this->networkWeights.size();

    // Each layer has its weights companions
    for (unsigned i = 0; i < size; i++) {
        std::cout << this->layersSizes[i] << "x" << 1 << " - Layer_" << i << std::endl;

        if (i == size - 1)
            break;

        std::cout << this->networkWeights[i].n_rows << "x" << this->networkWeights[i].n_cols << " - Weights_" << i << std::endl;
        if (showMatrices)
            std::cout << this->networkWeights[i] << std::endl;
    }

    std::cout << "/////////////////////// Layers and weights end //////////////////////////" << std::endl;
}

arma::Mat<double> neuralNework::NN::feedForward(arma::Mat<double>& input)
{
    unsigned i = 0;

    // Apply activation function
    // output = activationFunction(zMatrix)
    auto output = applyActivationFunc(
        this->networkWeights[i] * input, // Generate the zMatrix = (weights * input)
        i + 1 // Index of activation function
    );

    unsigned size = this->networkWeights.size();

    // Each layer has its weights companions except for the output layer
    for (i = 1; i < size; i++) {
        // Apply activation function
        // output = activationFunction(zMatrix)
        output = applyActivationFunc(
            this->networkWeights[i] * output, // Generate the zMatrix = (weights * input)
            i + 1 // Index of activation function
        );
    }

    return output;
}

inline arma::Mat<double> neuralNework::NN::applyActivationFunc(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->activationFunctions[index]);
}

inline arma::Mat<double> neuralNework::NN::applyActivationFuncD(arma::Mat<double> value, unsigned index)
{
    return value.transform(this->activationFunctionsD[index]);
}

void neuralNework::NN::backPropagation(arma::Mat<double>& target, arma::Mat<double>& input)
{

    float learnningRate = 0.01;

    ////////////////////
    /// Feed forward ///
    ////////////////////

    // Preparing to store the intermediate values
    const unsigned size = this->networkWeights.size();
    std::vector<arma::Mat<double>> layers(size);

    unsigned i = 0;
    // Apply activation function
    // output = activationFunction(zMatrix)
    layers[i] = applyActivationFunc(
        this->networkWeights[i] * input, // Generate the zMatrix = (weights * input)
        i + 1 // Index of activation function
    );

    // Each layer has its weights except for the output layer
    for (i = 1; i < size; i++) {
        // Apply activation function
        // output = activationFunction(zMatrix)
        layers[i] = applyActivationFunc(
            this->networkWeights[i] * layers[i - 1], // Generate the zMatrix = (weights * input)
            i + 1 // Index of activation function
        );
    }

    ///////////////////////
    /// Backpropagation ///
    ///////////////////////

    i--; // Points to output
    arma::Mat<double> errors = target - layers[i];

    // delta = gradient * hidden.tranposed
    arma::Mat<double> gradientHiddenToOutput = (this->applyActivationFuncD(layers[i - 1], i) % errors) * learnningRate;
    arma::Mat<double> deltaHiddenToOutput = gradientHiddenToOutput * layers[i].t();

    // Update the weights
    this->networkWeights[i] += deltaHiddenToOutput;
    // Update bias weights
    //    this->biasOutput += gradientHiddenToOutput;

    for (; i > 0; i++) {
        layers[i].transform(this->activationFunctionsD[i]);
        this->networkWeights[i] += (layers[i] % errors) * learnningRate * layers[i].t();
        errors = this->networkWeights[i].t() + layers[i];
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
