#include "lib/NN.h"

int main(int argc, char const *argv[])
{
    using namespace neuralNework;
    NN nn;
    nn.addLayer(2, LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(3, LayerType::FullConnected, ActivationFunction::Relu);
    nn.addLayer(4, LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(5, LayerType::FullConnected, ActivationFunction::Relu);
    nn.addLayer(4, LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(3, LayerType::FullConnected, ActivationFunction::Relu);
    nn.addLayer(2, LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(1, LayerType::Output, ActivationFunction::Sigmoid);
    nn.assemble();
    nn.feedForward();
    nn.showStructure(true);

    arma::Mat<double> target(1,1);
    target.at(0,0) = 1;

    nn.backPropagation(target);
    return 0;
}
