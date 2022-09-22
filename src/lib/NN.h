#ifndef SRC_LIB_NN_H
#define SRC_LIB_NN_H

#include <vector>
#include <armadillo>
#include "InputLayer.h"

namespace neuralNework
{
    double step(double x);
    double stepD(double x);

    double relu(double x);
    double reluD(double x);

    double silu(double x);
    double siluD(double x);

    double sigmoid(double x);
    double sigmoidD(double x);

    double softplus(double x);
    double softplusD(double x);

    double leakyRelu(double x);
    double leakyReluD(double x);

    double hiperbolicTangent(double x);
    double hiperbolicTangentD(double x);

    enum ActivationFunction
    {
        Relu,
        Step,
        Silu,
        Sigmoid,
        Hiperbolic,
        Softplus,
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
    std::vector<arma::Mat<double>> hiddenWeights,
    bool showMatrices = false);
    };
}
#endif // SRC_LIB_NN_H
