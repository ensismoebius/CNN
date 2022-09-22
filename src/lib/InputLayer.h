#ifndef SRC_LIB_INPUTLAYER_H
#define SRC_LIB_INPUTLAYER_H
#include <armadillo>
namespace neuralNework
{
    class InputLayer
    {
    public:
        arma::Mat<double> input;

    public:
        InputLayer(unsigned int length);
        ~InputLayer();
    };
}
#endif // SRC_LIB_INPUTLAYER_H
