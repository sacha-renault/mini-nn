#pragma once
#include <cmath>
#include "Value.hpp"

class ActivationFunction{
protected:
    virtual float _forward(float x) = 0;
    virtual float _backward(float x) = 0;
public:
    virtual Value operator()(Value& other) final {
        // TODO push_back in backward 
        float x = other.getData();
        float outVal = _forward(x);

        std::vector<Value*> childrens = std::vector<Value*> { &other };

        Value out = Value(
            outVal,
            childrens);

        out.setBackward([&out, &other, this]() {
            other.setGrad(_backward(out.getData()) * out.getGrad());
        });

        return out;
    }
};

class Tanh : public ActivationFunction {
protected:
    float _forward(float x) {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    }
    float _backward(float x) {
        return (1 - std::pow(x, 2));
    }
};