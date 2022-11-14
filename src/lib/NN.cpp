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

    // The first is just an input
    if (type == LayerType::Input)
        return true;

    switch (function) {
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
    case None:
        break;
    }

    return true;
}

bool neuralNework::NN::assemble()
{
    //////////////////////////////////////////////////////
    /////// The weights creating and initializing ////////
    //////////////////////////////////////////////////////
    const unsigned size = this->layersSizes.size();
    unsigned i;
    for (i = 0; i < size - 1; i++) {

        // TODO we need to allocate some space for biases too, that is why the +1 at the layersSizes[i] ??????
        // arma::Mat<double> hiddenWeight(this->layersSizes[i + 1], this->layersSizes[i] + 1);

        // The weights beetween previous (layersSizes[i]) and next (layersSizes[i+1]) layer
        arma::Mat<double> hiddenWeight(this->layersSizes[i + 1], this->layersSizes[i]);
        hiddenWeight.randu();

        // Biases for hidden layers
        arma::Mat<double> biases(this->layersSizes[i + 1], 1);
        biases.randu();

        // Adds the weights
        this->networkMatrices.push_back(hiddenWeight);

        // Adds the biases
        this->networkBiases.push_back(biases);
    }

    return true;
}

void neuralNework::NN::showStructure(bool showValues)
{
    std::cout << "//////////////////////// Layers, biases and weights ///////////////////////////" << std::endl;
    unsigned size = this->networkMatrices.size();
    unsigned i;

    // Each layer has its weights companions
    for (i = 0; i < size; i++) {
        std::cout << this->layersSizes[i] << "x" << 1 << " - Layer_" << i << std::endl;

        std::cout << this->networkMatrices[i].n_rows << "x" << this->networkMatrices[i].n_cols << " - Weights_" << i << std::endl;
        if (showValues)
            std::cout << this->networkMatrices[i] << std::endl;

        std::cout << "------------------------" << std::endl;
        // layersSizes[i + 1] comes from the fact that the input layer has no biases
        // the biases apply from the first hidden layer ahead
        std::cout << this->layersSizes[i + 1] << "x" << 1 << " - Biases_" << i + 1 << std::endl;
        if (showValues)
            std::cout << this->networkBiases[i] << std::endl;
    }
    std::cout << this->layersSizes[i] << "x" << 1 << " - Layer_" << i << std::endl;

    std::cout << "/////////////////////// Layers, biases and weights end //////////////////////////" << std::endl;
}

arma::Mat<double> neuralNework::NN::feedForward(arma::Mat<double>& input)
{
    // Generate the zMatrix = (weights * input + biases)
    arma::mat output = this->networkMatrices[0] * input + this->networkBiases[0];
    // Apply activation function
    // output = activationFunction(zMatrix)
    output.transform(this->activationFunctions[0]);

    // Each 2 layer interval has its weights
    // i=1 because the first interval is already
    // used at the code above
    unsigned size = this->networkMatrices.size();
    for (unsigned i = 1; i < size; i++) {
        // Generate the zMatrix = (weights * input + biases)
        output = this->networkMatrices[i] * output + this->networkBiases[i];
        // Apply activation function
        // output = activationFunction(zMatrix)
        output.transform(this->activationFunctions[i]);
    }

    return output;
}

void neuralNework::NN::backPropagation(arma::Mat<double>& target, arma::Mat<double>& input)
{
    // Calculates the first error
    static const arma::Mat<double> output = neuralNework::NN::feedForward(input);
    std::cout << output << std::endl;

    // Calculate the error for each output
    static arma::Mat<double> error = output - target;

    error.transform(this->errorFunction);

    //    // Sums the results of the error function applied in the error
    //    double sum = 0;
    //    double divisor = 0;
    //    arma::mat::iterator it_end = error.end();
    //    for (arma::mat::iterator it = error.begin(); it != it_end; ++it) {
    //        sum += this->errorFunction((*it));
    //        divisor++;
    //    }
    //    double totalError = sum / divisor;
    //    std::cout << totalError << std::endl;
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
               (1 + std::pow(M_E, -x)) + (x + std::pow(M_E, -x)))
        / (std::pow(1.0 + std::pow(M_E, -x), 2));
}
