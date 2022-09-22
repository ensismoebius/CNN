#include "NN.h"
#include <cmath>
#include <vector>
#include <limits>
#include <iostream>
#include <armadillo>
#include <algorithm>

neuralNework::NN::~NN()
{
}

neuralNework::NN::NN()
{
}

bool neuralNework::NN::addLayer(unsigned nodes, LayerType type, ActivationFunction function)
{
    this->layersTypes.push_back(type);
    this->layersSizes.push_back(nodes);

    if (type == LayerType::Output)
        return true;

    switch (function)
    {
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

bool neuralNework::NN::assemble(bool showStructure)
{
    /////////////////////////////////////////////////////////////
    /////// The hidden weights creating and initializing ////////
    /////////////////////////////////////////////////////////////
    for (unsigned i = 0; i < this->layersSizes.size(); i++)
    {
        //// Initialise all hidden layers
        arma::Mat<double> hiddenLayer(
            this->layersSizes[i],
            1);

        arma::Mat<double> hiddenWeight(
            this->layersSizes[i + 1],
            this->layersSizes[i]);
        hiddenWeight.randu();

        this->networkMatrices.push_back(hiddenLayer);
        this->networkMatrices.push_back(hiddenWeight);
    }

    this->showStructure(this->networkMatrices, showStructure);

    return true;
}

void neuralNework::NN::showStructure(
    std::vector<arma::Mat<double>> layers,
    bool showMatrices)
{

    std::cout << "//////////////////////// Layers and weights ///////////////////////////" << std::endl;
    for (int i = 0; i < layers.size(); i++)
    {
        std::cout << layers[i].n_rows << "x" << layers[i].n_cols << " - Layer_" << i << std::endl;
        if (showMatrices)
            std::cout << layers[i] << std::endl;

        i++;

        std::cout << layers[i].n_rows << "x" << layers[i].n_cols << " - Weights_" << i << std::endl;
        if (showMatrices)
            std::cout << layers[i] << std::endl;
    }
}

void neuralNework::NN::feedForward()
{
    unsigned size = this->networkMatrices.size() - 2;

    // Each layer has its weights companions
    for (unsigned i = 0; i < size; i += 2)
    {
        // Generate the zMatrix = (input * weights)
        this->networkMatrices[i + 2] = this->networkMatrices[i + 1] * this->networkMatrices[i];

        // Apply activation function
        this->networkMatrices[i + 2].transform(this->activationFunctions[i / 2]);
    }
}

void neuralNework::NN::backPropagation()
{
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
               (1 + std::pow(M_E, -x)) +
               (x + std::pow(M_E, -x))) /
           (std::pow(1.0 + std::pow(M_E, -x), 2));
}