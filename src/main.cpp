#include "lib/NN.h"

int main(int argc, char const *argv[])
{
    using namespace neuralNework;
    NN nn(2, 1);
    nn.addLayer(3 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(4 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(5 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(4 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(3 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.addLayer(2 , LayerType::FullConnected, ActivationFunction::Sigmoid);
    nn.assemble();

    return 0;
}
