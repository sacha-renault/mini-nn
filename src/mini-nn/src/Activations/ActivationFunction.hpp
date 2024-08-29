#pragma once
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../Values/Value.hpp"

namespace Activations {
    class BaseActivation { 
    public:
        virtual std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs)=0;
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

        virtual std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) final {
            std::vector<std::shared_ptr<Value>> outputs; 

            for (const auto& val : inputs) {
                outputs.push_back((*this)(val));
            }

            return outputs;
        }
    };

    class LambdaActivationVector : public BaseActivation {
    protected:
        std::function<std::vector<float>(const std::vector<float>& x)> _forward_func;
        std::function<std::vector<float>(const std::vector<float>& x)> _backward_func;

        virtual std::vector<float> _forward(const std::vector<float>& x) {
            return _forward_func(x);
        }
        virtual std::vector<float> _backward(const std::vector<float>& x) {
            return _backward_func(x);
        }

    public:
        LambdaActivationVector(
            std::function<std::vector<float>(const std::vector<float>& x)> forward,
            std::function<std::vector<float>(const std::vector<float>& x)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        virtual std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) override {
            // Convert input Values to raw float data
            std::vector<float> x(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                x[i] = inputs[i]->getData();
            }

            // Apply the forward function to compute the output
            std::vector<float> out = _forward_func(x);

            // Convert children to shared pointers
            std::vector<std::shared_ptr<Value>> children(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                children[i] = inputs[i];  // Use existing shared pointers to the input Values
            }

            // Initialize outputs with the correct children
            std::vector<std::shared_ptr<Value>> outputs;
            for (size_t i = 0; i < out.size(); ++i) {
                auto val = std::make_shared<Value>(out[i]);

                // Add children correctly
                for (const auto& child : children) {
                    val->addChild(child);
                }

                outputs.push_back(val);  // Store the output value
            }

            // Define the backward function for each output
            for (size_t i = 0; i < outputs.size(); ++i) {
                outputs[i]->setBackward([&, i]() {
                    // Compute the gradient with respect to the output
                    std::vector<float> grad_softmax = _backward_func(out);
                    for (size_t j = 0; j < inputs.size(); ++j) {
                        inputs[j]->accumulateGrad(grad_softmax[j] * outputs[i]->getGrad());
                    }
                });
            }

            // Return the outputs
            return outputs;
        }
    };
}
