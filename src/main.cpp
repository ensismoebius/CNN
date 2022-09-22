#include "lib/NN.h"

int main(int argc, char const *argv[])
{
    using namespace neuralNework;
    NN network(2, 2);

    network.assemble();

    return 0;
}
