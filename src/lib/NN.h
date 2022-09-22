#ifndef SRC_LIB_NN_H
#define SRC_LIB_NN_H

#include <vector>
#include <armadillo>
#include "InputLayer.h"

namespace neuralNework
{
    enum ActivationFunction
    {
        Relu,
        Sigmoid,
        LeakyRelu
    };

    enum LayerType
    {
        Input,
        Spike,
        Residual,
        FullConnected
    };

    class NN
    {
    private:
        unsigned inputSize;
        unsigned outputSize;
        std::vector<double> hiddenLayersSizes;
        std::vector<LayerType> hiddenLayersTypes;
        std::vector<ActivationFunction> hiddenLayersFunctions;

    public:
        ~NN();
        NN(unsigned inputSize, unsigned outputSize);
        bool addLayer(unsigned nodes, LayerType type, ActivationFunction function);
        bool assemble();
        void showStructure(
            arma::Mat<double> input,
            arma::Mat<double> inputWeights,
            arma::Mat<double> output,
            std::vector<arma::Mat<double>> hiddenLayers,
            std::vector<arma::Mat<double>> hiddenWeights);
    };
}
#endif // SRC_LIB_NN_H
