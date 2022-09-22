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

neuralNework::NN::NN(unsigned inputSize, unsigned outputSize) : inputSize(inputSize),
                                                                outputSize(outputSize)
{
}

bool neuralNework::NN::addLayer(unsigned nodes, LayerType type, ActivationFunction function)
{
    this->hiddenLayersTypes.push_back(type);
    this->hiddenLayersSizes.push_back(nodes);
    this->hiddenLayersFunctions.push_back(function);
    return true;
}

bool neuralNework::NN::assemble()
{
    // Input layer
    arma::Mat<double> input(this->inputSize, 1);

    // Output layer
    arma::Mat<double> output(this->outputSize, 1);

    // The input to hidden/output layer
    // weight matrix depends on how long
    // the input and the hidden layer/
    // output is
    unsigned int inputWeightsRows =
        this->hiddenLayersSizes.size() > 0 ? this->hiddenLayersSizes[0] : this->outputSize;

    /////////////////////////////////////////////////////////////
    /////////////////////// Input weights ///////////////////////
    /////////////////////////////////////////////////////////////
    arma::Mat<double> inputWeights(
        inputWeightsRows, // input weights rows
        this->inputSize   // input weights cols
    );
    inputWeights.randu(); // initialize ramdomly all weigths

    /////////////////////////////////////////////////////////////
    /////////////////////// The hidden layers ///////////////////
    /////////////////////////////////////////////////////////////
    std::vector<arma::Mat<double>> hiddenLayers;

    /////////////////////////////////////////////////////////////
    /////// The hidden weights creating and initializing ////////
    /////////////////////////////////////////////////////////////
    std::vector<arma::Mat<double>> hiddenWeights;
    for (unsigned i = 0; i < this->hiddenLayersSizes.size(); i++)
    {
        if (i == this->hiddenLayersSizes.size() - 1)
        {
            //// Initialise the last hidden to output
            arma::Mat<double> hiddenLayer(this->hiddenLayersSizes[i], 1);

            arma::Mat<double> hiddenWeight(
                this->outputSize,
                this->hiddenLayersSizes[i]);
            hiddenWeight.randu();

            hiddenLayers.push_back(hiddenLayer);
            hiddenWeights.push_back(hiddenWeight);
        }
        else
        {
            //// Initialise all hidden layers
            arma::Mat<double> hiddenLayer(
                this->hiddenLayersSizes[i],
                1);

            arma::Mat<double> hiddenWeight(
                this->hiddenLayersSizes[i + 1],
                this->hiddenLayersSizes[i]);
            hiddenWeight.randu();

            hiddenLayers.push_back(hiddenLayer);
            hiddenWeights.push_back(hiddenWeight);
        }
    }

    this->showStructure(input, inputWeights, output, hiddenLayers, hiddenWeights);

    return true;
}

void neuralNework::NN::showStructure(
    arma::Mat<double> input,
    arma::Mat<double> inputWeights,
    arma::Mat<double> output,
    std::vector<arma::Mat<double>> hiddenLayers,
    std::vector<arma::Mat<double>> hiddenWeights,
    bool showMatrices)
{

    std::cout << "//////////////////////// Layers ///////////////////////////" << std::endl;
    std::cout << "Input  layer   - \t" << input.n_rows << "x" << input.n_cols << std::endl;
    if (showMatrices)
        std::cout << input << std::endl;

    for (unsigned int i = 0; i < hiddenLayers.size(); i++)
    {
        std::cout << "Hidden layer " << i << " - \t" << hiddenLayers[i].n_rows << "x" << hiddenLayers[i].n_cols << std::endl;
        if (showMatrices)
            std::cout << hiddenLayers[i] << std::endl;
    }

    std::cout << "Output layer   - \t" << output.n_rows << "x" << output.n_cols << std::endl;
    if (showMatrices)
        std::cout << output << std::endl;

    std::cout << "//////////////////////// Weights ///////////////////////////" << std::endl;
    std::cout << "Input  weight    - \t" << inputWeights.n_rows << "x" << inputWeights.n_cols << std::endl;
    if (showMatrices)
        std::cout << inputWeights << std::endl;

    for (unsigned int i = 0; i < hiddenLayers.size(); i++)
    {
        std::cout << "Hidden weights " << i << " - \t" << hiddenWeights[i].n_rows << "x" << hiddenWeights[i].n_cols << std::endl;
        if (showMatrices)
            std::cout << hiddenWeights[i] << std::endl;
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