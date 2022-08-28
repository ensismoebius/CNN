#ifndef LAYER_H
#define LAYER_H

#pragma once

#include <vector>
#include <Eigen/Core>
#include "../utils/Rng.hpp"
#include "../utils/Config.hpp"
#include "../optimizer/Optimizer.hpp"

class Layer
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    const int M_IN_SIZE;
    const int M_OUT_SIZE;

    Layer(const int inputSize, const int outputSize);
    virtual ~Layer();

    virtual const Matrix output() = 0;
    virtual const Matrix &backpropData () const = 0;

    virtual void backprop(const Matrix &preLayerOutput, const Matrix &nextLayerData) = 0;

    virtual void init(const Scalar &mu, const Scalar &sigma, Rng &rng) = 0;
    virtual void forward(const Matrix &prevLayerOutput) = 0;

    virtual std::vector<Scalar> getParameter() const = 0;
    virtual void setParameter(const std::vector<Scalar> &param);
    virtual std::vector<Scalar> getDerivatives() const = 0;
    






    
private:

};

#endif