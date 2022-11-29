#ifndef SRC_LIB_NN_H
#define SRC_LIB_NN_H

#include <vector>
#include <armadillo>

namespace neuralNework
{
    double none(double x);
    double noneD(double x);

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
        LeakyRelu,
        None
    };

    double absError(double error);
    double simpleError(double error);
    double quadraticError(double error);

    enum ErrorFunction
    {
        SimpleError,
        AbsoluteError,
        QuadraticError
    };

    enum LayerType
    {
        Input,
        Spike,
        Residual,
        FullConnected,
        Output
    };

    class NN
    {
    private:
        std::vector<double> layersSizes;
        std::vector<LayerType> layersTypes;
        std::vector<double (*)(double)> activationFunctions;
        std::vector<double (*)(double)> activationFunctionsD;

        double (*errorFunction)(double error);

        // Assembled neural network
        std::vector<arma::Mat<double>> networkWeights;

    public:
        NN(ErrorFunction errorFunction);
        ~NN();

        arma::Mat<double> applyActivationFunc(arma::Mat<double> value, unsigned index);
        arma::Mat<double> applyActivationFuncD(arma::Mat<double> value, unsigned index);

        bool assemble();
        void showStructure(bool showMatrices = false);
        void backPropagation(arma::Mat<double> &target, arma::Mat<double> &input);
        bool addLayer(unsigned nodes, LayerType type, ActivationFunction function = None);

        arma::Mat<double> feedForward(arma::Mat<double> &input);
    };
}
#endif // SRC_LIB_NN_H
