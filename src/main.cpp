#include "lib/NN.h"
#include <iostream>

int main(int argc, char const* argv[])
{
    using namespace neuralNework;
    NN nn(ErrorFunction::QuadraticError);

    nn.addLayer(2, LayerType::Input);
    nn.addLayer(2, LayerType::FullConnected, ActivationFunction::Relu);
    nn.addLayer(1, LayerType::Output, ActivationFunction::Hiperbolic);

    arma::Mat<double> target(1, 1);
    target.at(0, 0) = 0;

    arma::Mat<double> input(2, 1);
    input.at(0, 0) = 1;
    input.at(1, 0) = 1;

    nn.assemble();
    nn.feedForward(input);
    nn.showStructure(true); // TODO Corrigir para mostrar tudo corretamente
    nn.backPropagation(target, input);
    return 0;
}
