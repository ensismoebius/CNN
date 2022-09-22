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
        FullConnected,
        Residual,
        Spike
    };

    class NN
    {
    private:
        unsigned int inputSize;
        std::vector<double> hiddenLayersSizes;
        std::vector<LayerType> hiddenLayersTypes;
        std::vector<ActivationFunction> hiddenLayersFunctions;
        unsigned int outputSize;

    public:
        NN(unsigned int inputSize, unsigned int outputSize);
        ~NN();
        bool addLayer(unsigned int nodes, LayerType type, ActivationFunction function);
        bool assemble();
    };
}
#endif // SRC_LIB_NN_H
