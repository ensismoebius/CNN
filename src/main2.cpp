#include <armadillo>
#include <cmath>
#include <iostream>

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

    double learnningRate = 0.5;
    std::cout << "///////////// Learnning rate /////////////" << std::endl;
    std::cout << learnningRate << std::endl
              << std::endl;

    arma::Mat<double> target(2, 1);
    target.at(0, 0) = 0.01;
    target.at(1, 0) = 0.99;
    std::cout << "///////////// Target /////////////" << std::endl;
    std::cout << target << std::endl;

    arma::Mat<double> layer0(2, 1);
    layer0.at(0, 0) = 0.05;
    layer0.at(1, 0) = 0.10;
    std::cout << "///////////// Layer 0 (Input) /////////////" << std::endl;
    std::cout << layer0 << std::endl;

    arma::Mat<double> weight0to1(2, 2);
    weight0to1.at(0, 0) = 0.15;
    weight0to1.at(0, 1) = 0.20;
    weight0to1.at(1, 0) = 0.25;
    weight0to1.at(1, 1) = 0.30;
    std::cout << "///////////// weight0 (InputHiddenWeights) /////////////" << std::endl;
    std::cout << weight0to1 << std::endl;

    arma::Mat<double> bias0(2, 1);
    bias0.at(0, 0) = 0.35;
    bias0.at(1, 0) = 0.35;

    arma::Mat<double> layer1(2, 1);
    std::cout << "///////////// Layer 1 (Hidden) /////////////" << std::endl;
    std::cout << layer1 << std::endl;

    arma::Mat<double> weight1to2(2, 2);
    weight1to2.at(0, 0) = 0.4;
    weight1to2.at(0, 1) = 0.45;
    weight1to2.at(1, 0) = 0.50;
    weight1to2.at(1, 1) = 0.55;
    std::cout << "///////////// weight1 (HiddenOutputWeights) /////////////" << std::endl;
    std::cout << weight1to2 << std::endl;

    arma::Mat<double> bias1(2, 1);
    bias1.at(0, 0) = 0.60;
    bias1.at(1, 0) = 0.60;

    arma::Mat<double> layer2(2, 1);
    std::cout << "///////////// Layer 2 (Output) /////////////" << std::endl;
    std::cout << layer2 << std::endl;

    std::cout << "/////////////////////////////////////////////" << std::endl;

    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    std::cout << "------------ FeedForward: BEGIN" << std::endl;

    std::cout << " FeedForward: Layer 0 (input)" << std::endl;
    std::cout << layer0 << std::endl;

    std::cout << " FeedForward: Layer 1 (hidden)" << std::endl;
    layer1 = weight0to1 * layer0 + bias0;
    layer1 = applyActivationFunc(layer1);
    std::cout << layer1 << std::endl;

    std::cout << " FeedForward: Layer 2 (output)" << std::endl;
    layer2 = weight1to2 * layer1 + bias1;
    layer2 = applyActivationFunc(layer2);
    std::cout << layer2 << std::endl;

    std::cout << " FeedForward: Error" << std::endl;
    arma::Mat<double> errors = ((target - layer2) % (target - layer2)) / 2;
    std::cout << errors << std::endl;

    std::cout << "------------ FeedForward: END" << std::endl;
    std::cout << "------------ Backprop OUTPUT: BEGIN" << std::endl;

    std::cout << "a- Backprop OUTPUT: Layer 2 errors derivative (output errors derivative)" << std::endl;
    // this is the derivative of the error with respect to the last layer (aka output)
    arma::Mat<double> errorSlope = layer2 - target;
    std::cout << errorSlope << std::endl;

    std::cout << "b- Backprop OUTPUT: Layer 2 activation function derivative" << std::endl;
    // this is the derivative of activation function with respect with its input
    // (the z matrix)
    auto activationSlope2 = applyActivationFuncD(layer2);
    std::cout << activationSlope2 << std::endl;

    std::cout << "c- Backprop OUTPUT: Layer 2 Zmatrix derivative (Layer 1 Transposed)" << std::endl;
    // this is the derivative of the Z matrix with respect with
    // his weights (aka the value of the layer 1).
    std::cout << layer1.t() << std::endl;

    std::cout << "d- Backprop OUTPUT: Layer 2 delta (a*b)" << std::endl;
    auto delta2 = errorSlope % activationSlope2;
    std::cout << delta2 << std::endl;

    std::cout << " Backprop OUTPUT: Layer 2 gradient (c*d)" << std::endl;
    // this is the derivative of the error with respect
    // to the weights
    auto gradient2 = delta2 * layer1.t();
    std::cout << gradient2 << std::endl;

    std::cout << " Backprop OUTPUT: New Weight 1 (hidden->output)" << std::endl;
    arma::Mat<double> weight1to2Prime = weight1to2 - learnningRate * gradient2;
    std::cout << weight1to2 << "to" << std::endl
              << weight1to2Prime << std::endl;

    std::cout << "------------ Backprop OUTPUT: END" << std::endl;
    std::cout << "------------ Backprop HIDDEN: BEGIN" << std::endl;

    std::cout << "a- Backprop HIDDEN: Layer 1 activation function derivative" << std::endl;
    // this is the derivative of activation function with respect with its input
    // (the z matrix)
    auto activationSlope1 = applyActivationFuncD(layer1);
    std::cout << activationSlope1 << std::endl;

    std::cout << "b- Backprop HIDDEN: Layer 1 delta" << std::endl;
    auto delta1 = weight1to2.t() * delta2 % activationSlope1;
    std::cout << delta1 << std::endl;

    std::cout << "c- Backprop HIDDEN: Layer 1 gradient" << std::endl;
    // this is the derivative of the error with respect to the weights
    auto gradient1 = delta1 * layer0.t();
    std::cout << gradient1 << std::endl;

    std::cout << "d- Backprop HIDDEN: New Weight 0 (input->hidden)" << std::endl;
    arma::Mat<double> weight0to1Prime = weight0to1 - learnningRate * gradient1;
    std::cout << weight0to1 << "to" << std::endl
              << weight0to1Prime << std::endl;

    std::cout << "------------ Backprop HIDDEN: END" << std::endl;

    return 0;
}
