#include "NN.h"
#include <armadillo>
#include <vector>
#include <iostream>

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
    std::vector<arma::Mat<double>> hiddenWeights)
{
    std::cout << "Input - " << input.n_rows << "x" << input.n_cols << std::endl;
    std::cout << input << std::endl;
    std::cout << "Input weight - " << inputWeights.n_rows << "x" << inputWeights.n_cols << std::endl;
    std::cout << inputWeights << std::endl;

    for (unsigned int i = 0; i < hiddenLayers.size(); i++)
    {
        std::cout << "Hidden layer " << i << " - " << hiddenLayers[i].n_rows << "x" << hiddenLayers[i].n_cols << std::endl;
        std::cout << hiddenLayers[i] << std::endl;
        std::cout << "Hidden weights " << i << " - " << hiddenWeights[i].n_rows << "x" << hiddenWeights[i].n_cols << std::endl;
        std::cout << hiddenWeights[i] << std::endl;
    }

    std::cout << "Output layer - " << output.n_rows << "x" << output.n_cols << std::endl;
    std::cout << output << std::endl;
}
