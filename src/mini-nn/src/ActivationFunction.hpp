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

class LambdaActivation : public ActivationFunction {
protected:
    std::function<float(float)> _forward_func;
    std::function<float(float)> _backward_func;

    float _forward(float x) {
        return _forward_func(x);
    }
    float _backward(float x) {
        return _backward_func(x);
    } 

public:
    LambdaActivation(std::function<float(float)> forward, std::function<float(float)> backward)
        : _forward_func(forward), _backward_func(backward) { }
};