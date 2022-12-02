#include "lib/NN.h"
#include <iostream>

// int main(int argc, char const* argv[])
//{
//     using namespace neuralNework;
//     NN nn(ErrorFunction::QuadraticError);

//    nn.addLayer(2, LayerType::Input);
//    nn.addLayer(2, LayerType::FullConnected, ActivationFunction::Relu);
//    nn.addLayer(1, LayerType::Output, ActivationFunction::Hiperbolic);

//    arma::Mat<double> target(1, 1);
//    target.at(0, 0) = 0;

//    arma::Mat<double> input(2, 1);
//    input.at(0, 0) = 1;
//    input.at(1, 0) = 1;

//    nn.assemble();
//    nn.feedForward(input);
//    nn.showStructure(true); // TODO Corrigir para mostrar tudo corretamente
//    nn.backPropagation(target, input);
//    return 0;
//}
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::pow(M_E, -x));
}

double sigmoidD(double y)
{
    return y * (1.0 - y);
}

inline arma::Mat<double> applyActivationFuncD(arma::Mat<double> value)
{
    return value.transform(sigmoidD);
}

inline arma::Mat<double> applyActivationFunc(arma::Mat<double> value)
{
    return value.transform(sigmoid);
}

int main(int argc, char* argv[])
{
    arma::Mat<double> target(2, 1);
    target.at(0, 0) = 0.01;
    target.at(1, 0) = 0.99;

    arma::Mat<double> layer0(2, 1);
    layer0.at(0, 0) = 0.05;
    layer0.at(1, 0) = 0.10;
    std::cout << layer0 << std::endl;

    arma::Mat<double> weight0(2, 2);
    weight0.at(0, 0) = 0.15;
    weight0.at(0, 1) = 0.20;
    weight0.at(1, 0) = 0.25;
    weight0.at(1, 1) = 0.30;
    std::cout << weight0 << std::endl;

    arma::Mat<double> layer1(2, 1);
    std::cout << layer1 << std::endl;
    arma::Mat<double> bias0(2, 1);
    bias0.at(0, 0) = 0.35;
    bias0.at(1, 0) = 0.35;

    arma::Mat<double> weight1(2, 2);
    weight1.at(0, 0) = 0.4;
    weight1.at(0, 1) = 0.45;
    weight1.at(1, 0) = 0.50;
    weight1.at(1, 1) = 0.55;
    std::cout << weight1 << std::endl;

    arma::Mat<double> layer2(2, 1);
    std::cout << layer2 << std::endl;
    arma::Mat<double> bias1(2, 1);
    bias1.at(0, 0) = 0.60;
    bias1.at(1, 0) = 0.60;

    std::cout << "------------FeedForward->input->hidden--------------" << std::endl;
    layer1 = weight0 * layer0 + bias0;
    layer1 = applyActivationFunc(layer1);
    std::cout << layer1 << std::endl;

    std::cout << "------------FeedForward->hidden->output--------------" << std::endl;
    layer2 = weight1 * layer1 + bias1;
    layer2 = applyActivationFunc(layer2);
    std::cout << layer2 << std::endl;

    std::cout << "------------Errors--------------" << std::endl;

    arma::Mat<double> errors = ((target - layer2) % (target - layer2)) / 2;
    std::cout << errors << std::endl;

    std::cout << "------------Errors->totalError--------------" << std::endl;
    double totalError = 0;
    for (const auto& v : errors) {
        totalError += v;
    }
    std::cout << totalError << std::endl;

    std::cout << "------------Backprop->errors->errorSlope--------------" << std::endl;

    arma::Mat<double> errorSlope = layer2 - target;
    std::cout << errorSlope << std::endl;

    std::cout << "------------Backprop->errors->activationSlope--------------" << std::endl;

    auto activationSlope = applyActivationFuncD(layer2);
    std::cout << activationSlope << std::endl;

    std::cout << "------------Backprop->errors->weightSlope--------------" << std::endl;
    std::cout << layer1 << std::endl;

    std::cout << "------------Backprop->errors->delta1--------------" << std::endl;

    //    auto delta1 = errorSlope * totalErrorSlope * layer1.t();
    //    std::cout << delta1 << std::endl;

    return 0;
}
