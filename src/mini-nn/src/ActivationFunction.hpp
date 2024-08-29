#pragma once
#include <cmath>
#include "Value.hpp"

namespace Activations {
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

    // ReLU activation function
    static LambdaActivation ReLU(
        [](float x) { return std::max(0.0f, x); },           // Forward: ReLU
        [](float x) { return x > 0.0f ? 1.0f : 0.0f; }       // Backward: ReLU derivative
    );

    // Tanh activation function
    static LambdaActivation Tanh(
        [](float x) { return std::tanh(x); },                // Forward: Tanh
        [](float x) { float tanhx = std::tanh(x); return 1.0f - tanhx * tanhx; }  // Backward: Tanh derivative
    );

    // Sigmoid activation function
    static LambdaActivation Sigmoid(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },  // Forward: Sigmoid
        [](float x) { float sigx = 1.0f / (1.0f + std::exp(-x)); return sigx * (1.0f - sigx); }  // Backward: Sigmoid derivative
    );
}