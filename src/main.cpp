#include "lib/NN.h"
#include <iostream>

int main(int argc, char const* argv[])
{
    using namespace neuralNework;
    NN nn(ErrorFunction::QuadraticError);

    nn.addLayer(2, LayerType::Input);
    nn.addLayer(2, LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(2, LayerType::Output, ActivationFunction::Sigmoid);

    arma::Mat<double> target(2, 1);
    target.at(0, 0) = 0.01;
    target.at(1, 0) = 0.99;

    arma::Mat<double> input(2, 1);
    input.at(0, 0) = 0.05;
    input.at(1, 0) = 0.10;

    nn.assemble();
    //    nn.feedForward(input);
    //    nn.showStructure(true); // TODO Corrigir para mostrar tudo corretamente
    nn.backPropagation(target, input);
    return 0;
}
