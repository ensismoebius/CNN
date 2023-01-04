#include "lib/NN.h"
#include <cstdio>
#include <iostream>
#include <vector>

struct trainningSample {
    arma::Mat<double> input;
    arma::Mat<double> target;
};

void populateExamples(std::vector<trainningSample>& examples)
{
    examples.resize(19);

    examples[0].input = arma::Mat<double>(2, 1);
    examples[0].input.at(0, 0) = 0;
    examples[0].input.at(1, 0) = 0;
    examples[0].target = arma::Mat<double>(1, 1);
    examples[0].target.at(0, 0) = 0;

    examples[1].input = arma::Mat<double>(2, 1);
    examples[1].input.at(0, 0) = 0;
    examples[1].input.at(1, 0) = 1;
    examples[1].target = arma::Mat<double>(1, 1);
    examples[1].target.at(0, 0) = 1;

    examples[2].input = arma::Mat<double>(2, 1);
    examples[2].input.at(0, 0) = 1;
    examples[2].input.at(1, 0) = 0;
    examples[2].target = arma::Mat<double>(1, 1);
    examples[2].target.at(0, 0) = 1;

    examples[3].input = arma::Mat<double>(2, 1);
    examples[3].input.at(0, 0) = 1;
    examples[3].input.at(1, 0) = 1;
    examples[3].target = arma::Mat<double>(1, 1);
    examples[3].target.at(0, 0) = 0;

    examples[4].input = arma::Mat<double>(2, 1);
    examples[4].input.at(0, 0) = 4;
    examples[4].input.at(1, 0) = 4;
    examples[4].target = arma::Mat<double>(1, 1);
    examples[4].target.at(0, 0) = 0;

    examples[5].input = arma::Mat<double>(2, 1);
    examples[5].input.at(0, 0) = 7;
    examples[5].input.at(1, 0) = 0;
    examples[5].target = arma::Mat<double>(1, 1);
    examples[5].target.at(0, 0) = 1;

    examples[6].input = arma::Mat<double>(2, 1);
    examples[6].input.at(0, 0) = 0;
    examples[6].input.at(1, 0) = 7;
    examples[6].target = arma::Mat<double>(1, 1);
    examples[6].target.at(0, 0) = 1;

    examples[7].input = arma::Mat<double>(2, 1);
    examples[7].input.at(0, 0) = 3;
    examples[7].input.at(1, 0) = 2;
    examples[7].target = arma::Mat<double>(1, 1);
    examples[7].target.at(0, 0) = 1;

    examples[8].input = arma::Mat<double>(2, 1);
    examples[8].input.at(0, 0) = 2;
    examples[8].input.at(1, 0) = 3;
    examples[8].target = arma::Mat<double>(1, 1);
    examples[8].target.at(0, 0) = 1;

    examples[9].input = arma::Mat<double>(2, 1);
    examples[9].input.at(0, 0) = 6;
    examples[9].input.at(1, 0) = 7;
    examples[9].target = arma::Mat<double>(1, 1);
    examples[9].target.at(0, 0) = 1;

    examples[10].input = arma::Mat<double>(2, 1);
    examples[10].input.at(0, 0) = 2;
    examples[10].input.at(1, 0) = 1;
    examples[10].target = arma::Mat<double>(1, 1);
    examples[10].target.at(0, 0) = 1;

    examples[11].input = arma::Mat<double>(2, 1);
    examples[11].input.at(0, 0) = 1;
    examples[11].input.at(1, 0) = 2;
    examples[11].target = arma::Mat<double>(1, 1);
    examples[11].target.at(0, 0) = 1;

    examples[12].input = arma::Mat<double>(2, 1);
    examples[12].input.at(0, 0) = 7;
    examples[12].input.at(1, 0) = 6;
    examples[12].target = arma::Mat<double>(1, 1);
    examples[12].target.at(0, 0) = 1;

    examples[13].input = arma::Mat<double>(2, 1);
    examples[13].input.at(0, 0) = 0;
    examples[13].input.at(1, 0) = 2;
    examples[13].target = arma::Mat<double>(1, 1);
    examples[13].target.at(0, 0) = 1;

    examples[14].input = arma::Mat<double>(2, 1);
    examples[14].input.at(0, 0) = 2;
    examples[14].input.at(1, 0) = 0;
    examples[14].target = arma::Mat<double>(1, 1);
    examples[14].target.at(0, 0) = 1;

    examples[15].input = arma::Mat<double>(2, 1);
    examples[15].input.at(0, 0) = 0;
    examples[15].input.at(1, 0) = 4;
    examples[15].target = arma::Mat<double>(1, 1);
    examples[15].target.at(0, 0) = 1;

    examples[16].input = arma::Mat<double>(2, 1);
    examples[16].input.at(0, 0) = 4;
    examples[16].input.at(1, 0) = 0;
    examples[16].target = arma::Mat<double>(1, 1);
    examples[16].target.at(0, 0) = 1;

    examples[17].input = arma::Mat<double>(2, 1);
    examples[17].input.at(0, 0) = 15;
    examples[17].input.at(1, 0) = 15;
    examples[17].target = arma::Mat<double>(1, 1);
    examples[17].target.at(0, 0) = 0;

    examples[18].input = arma::Mat<double>(2, 1);
    examples[18].input.at(0, 0) = 7;
    examples[18].input.at(1, 0) = 9;
    examples[18].target = arma::Mat<double>(1, 1);
    examples[18].target.at(0, 0) = 1;
}

int main(int argc, char const* argv[])
{
    using namespace neuralNework;
    NN nn(ErrorFunction::QuadraticError);

    nn.addLayer(2, LayerType::Input);
    nn.addLayer(3, LayerType::FullConnected, ActivationFunction::Relu);
    nn.addLayer(1, LayerType::Output, ActivationFunction::Relu);

    std::vector<trainningSample> t;

    populateExamples(t);

    //    arma::Mat<double> target(2, 1);
    //    target.at(0, 0) = 0;
    //    target.at(1, 0) = 1;

    //    arma::Mat<double> input(2, 1);
    //    input.at(0, 0) = 0.25;
    //    input.at(1, 0) = 0.02;

    nn.assemble();

    for (int i = 0; i < 100000; i++) {
        for (auto sample : t) {
            nn.backPropagation(sample.target, sample.input, 0.5);
        }
    }

    nn.feedForward(t[0].input);
    nn.showStructure(true);

    nn.feedForward(t[1].input);
    nn.showStructure(true);

    return 0;
}
