#include "lib/NN.h"

int main(int argc, char const *argv[])
{
    using namespace neuralNework;
    NN network(2, 1);
    network.addLayer(
        4,
        LayerType::FullConnected,
        ActivationFunction::Sigmoid);
    network.addLayer(
        3,
        LayerType::FullConnected,
        ActivationFunction::Sigmoid);

    network.addLayer(
        10,
        LayerType::FullConnected,
        ActivationFunction::Sigmoid);

    network.addLayer(
        10,
        LayerType::FullConnected,
        ActivationFunction::Sigmoid);

    network.addLayer(
        5,
        LayerType::FullConnected,
        ActivationFunction::Sigmoid);

    network.assemble();

    return 0;
}
