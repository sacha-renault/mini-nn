#pragma once
#include <cmath>
#include <algorithm>
#include <numeric>
#include "Value.hpp"

namespace Activations {
    class BaseActivation { 
    public:
        virtual std::vector<Value> operator()(std::vector<Value>& inputs) = 0;
    };


    class LambdaActivation : public BaseActivation {
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

        virtual std::vector<Value> operator()(std::vector<Value>& inputs) final {
            std::vector<Value> outputs; 

            for (auto &val : inputs) {
                outputs.push_back((*this)(val));
            }

            return outputs;
        }
    };

    class ActivationFunctionVectorLambda : public BaseActivation{
    protected:
        std::function<std::vector<float>(const std::vector<float>& x)> _forward_func;
        std::function<std::vector<float>(const std::vector<float>& x)> _backward_func;

        virtual std::vector<float> _forward(const std::vector<float>& x){
            return _forward_func(x);
        };
        virtual std::vector<float> _backward(const std::vector<float>& x){
            return _backward_func(x);
        };

    public:
        ActivationFunctionVectorLambda(std::function<std::vector<float>(const std::vector<float>& x)> forward, std::function<std::vector<float>(const std::vector<float>& x)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        std::vector<Value> operator()(std::vector<Value>& inputs) {
            std::vector<float> x(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                x[i] = inputs[i].getData();
            }

            std::vector<float> softmax_out = _forward(x);

            std::vector<Value*> children(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                children[i] = &inputs[i];
            }

            std::vector<Value> outputs;
            for (size_t i = 0; i < softmax_out.size(); ++i) {
                outputs.emplace_back(Value(softmax_out[i], children));
            }

            for (size_t i = 0; i < outputs.size(); ++i) {
                outputs[i].setBackward([&, i]() {
                    std::vector<float> grad_softmax = _backward(softmax_out);
                    for (size_t j = 0; j < inputs.size(); ++j) {
                        inputs[j].setGrad(grad_softmax[j] * outputs[i].getGrad());
                    }
                });
            }

            return outputs;
        }
    };

    // ReLU activation function
    static LambdaActivation ReLU(
        [](float x) { return std::max(0.0f, x); },           // Forward: ReLU
        [](float x) { return x > 0.0f ? 1.0f : 0.0f; }       // Backward: ReLU derivative
    );

    // Tanh activation function
    static LambdaActivation Tanh(
        [](float x) { return std::tanh(x); },                // Forward: Tanh
        [](float x) { return 1.0f - x * x; }  // Backward: Tanh derivative
    );

    // Sigmoid activation function
    static LambdaActivation Sigmoid(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },  // Forward: Sigmoid
        [](float x) { return x * (1.0f - x); }  // Backward: Sigmoid derivative
    );

    // Softmax activation function
    static ActivationFunctionVectorLambda Softmax(
        // Forward function
        [](const std::vector<float>& x) -> std::vector<float> {
            std::vector<float> exp_x(x.size());
            float max_val = *std::max_element(x.begin(), x.end());
            
            // Compute exponentials for numerical stability
            for (size_t i = 0; i < x.size(); ++i) {
                exp_x[i] = std::exp(x[i] - max_val);
            }
            
            float sum_exp_x = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f);

            // Normalize to get probabilities
            for (size_t i = 0; i < exp_x.size(); ++i) {
                exp_x[i] /= sum_exp_x;
            }

            return exp_x;
        },
        
        // Backward function
        [](const std::vector<float>& softmax_output) -> std::vector<float> {
            std::vector<float> grad(softmax_output.size());
            // Compute the gradient for each output using the simplified diagonal Jacobian form
            for (size_t i = 0; i < softmax_output.size(); ++i) {
                grad[i] = softmax_output[i] * (1.0f - softmax_output[i]);
            }
            return grad;
        }
    );
}