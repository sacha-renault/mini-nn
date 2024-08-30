#pragma once
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include "../Values/Value.hpp"
#include "../Values/Tensor.hpp"

namespace Activations {

    class BaseActivation { 
    public:
        virtual Tensor1D operator()(const Tensor1D& inputs) = 0;
    };

    class ElementWiseActivation : public BaseActivation {
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
        ElementWiseActivation(std::function<float(float)> forward, std::function<float(float)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        virtual std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& input) final {
            float x = input->getData();
            float outVal = _forward(x);

            auto out = std::make_shared<Value>(outVal);
            out->addChild(input);

            out->setBackward([out, input, this]() {
                input->accumulateGrad(_backward(out->getData()) * out->getGrad());
            });

            return out;
        }

        virtual Tensor1D operator()(const Tensor1D& inputs) override {
            Tensor1D outputs(inputs.dimensions());  // Create a tensor with the same dimensions as inputs

            for (int i = 0; i < inputs.size(); ++i) {
                outputs({i}) = this->operator()(inputs({i}));  // Apply the forward operation on each element
            }

            return outputs;
        }
    };

    class TensorWiseActivation : public BaseActivation {
    protected:
        std::function<std::vector<float>(const std::vector<float>& x)> _forward_func;
        std::function<std::vector<float>(const std::vector<float>& x)> _backward_func;

        std::vector<float> _forward(const std::vector<float>& x) {
            return _forward_func(x);
        }

        std::vector<float> _backward(const std::vector<float>& x) {
            return _backward_func(x);
        }

    public:
        TensorWiseActivation(
            std::function<std::vector<float>(const std::vector<float>& x)> forward,
            std::function<std::vector<float>(const std::vector<float>& x)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        virtual Tensor1D operator()(const Tensor1D& inputs) override {
            // Convert input Values to raw float data
            std::vector<float> x(inputs.size());
            for (int i = 0; i < inputs.size(); ++i) {
                x[i] = inputs({i})->getData();
            }

            // Apply the forward function to compute the output
            std::vector<float> out = _forward_func(x);

            // Initialize outputs
            Tensor1D outputs(inputs.dimensions());
            for (int i = 0; i < out.size(); ++i) {
                auto val = std::make_shared<Value>(out[i]);

                // Add the original inputs as children for backpropagation
                for (int j = 0; j < inputs.size(); ++j) {
                    val->addChild(inputs({j}));
                }

                outputs({i}) = val;
            }

            // Define the backward function for each output element
            for (int i = 0; i < outputs.size(); ++i) {
                outputs({i})->setBackward([inputs, outputs, this, i, x]() {
                    std::vector<float> grad_softmax = _backward_func(x);
                    for (int j = 0; j < inputs.size(); ++j) {
                        inputs({j})->accumulateGrad(grad_softmax[j] * outputs({i})->getGrad());
                    }
                });
            }

            return outputs;
        }
    };
}
