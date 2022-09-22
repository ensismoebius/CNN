#include "NN.h"
#include <armadillo>
#include <vector>
#include <iostream>

neuralNework::NN::NN(unsigned int inputSize, unsigned int outputSize) : inputSize(inputSize),
                                                                        outputSize(outputSize)
{
}

neuralNework::NN::~NN()
{
}

bool neuralNework::NN::addLayer(unsigned int nodes, LayerType type, ActivationFunction function)
{
    this->hiddenLayersTypes.push_back(type);
    this->hiddenLayersSizes.push_back(nodes);
    this->hiddenLayersFunctions.push_back(function);
    return true;
}

bool neuralNework::NN::assemble()
{
    // Output layer
    arma::Mat<double> output(this->outputSize, 1);

    // Input layer
    arma::Mat<double> input(this->inputSize, 1);

    // The input to hidden/output layer
    // weight matrix depends on how long
    // the input and the hidden layer/
    // output is
    unsigned int inputWeightsRows =
        this->hiddenLayersSizes.size() > 0 ? this->hiddenLayersSizes[0] : this->outputSize;

    // Input weights
    arma::Mat<double> inputWeights(
        inputWeightsRows,
        this->inputSize // inputWeightsCols
    );

    // The hidden layers
    std::vector<arma::Mat<double>> hiddenLayers;
    // The hidden weights
    std::vector<arma::Mat<double>> hiddenWeights;

    // Assemble the hidden layers
    for (
        unsigned int i = 0;
        i < this->hiddenLayersSizes.size();
        i++)
    {

        if (i == this->hiddenLayersSizes.size() - 1)
        {
            arma::Mat<double> hiddenLayer(
                this->hiddenLayersSizes[i],
                1);

            arma::Mat<double> hiddenWeight(
                this->outputSize,
                this->hiddenLayersSizes[i]);

            hiddenLayers.push_back(hiddenLayer);
            hiddenWeights.push_back(hiddenWeight);
        }
        else
        {
            arma::Mat<double> hiddenLayer(
                this->hiddenLayersSizes[i],
                1);

            arma::Mat<double> hiddenWeight(
                this->hiddenLayersSizes[i + 1],
                this->hiddenLayersSizes[i]);

            hiddenLayers.push_back(hiddenLayer);
            hiddenWeights.push_back(hiddenWeight);
        }
    }

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

    return true;
}
