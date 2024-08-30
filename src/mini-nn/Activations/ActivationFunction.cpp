#include "ActivationFunction.hpp"

namespace Activations {
    /// @brief Activation on a single neuron
    /// @param input value (taht is neuron output)
    /// @return output tensor 
    std::shared_ptr<Value> ElementWiseActivation::operator()(const std::shared_ptr<Value>& input){
        float x = input->getData();
        float outVal = _forward(x);

        auto out = std::make_shared<Value>(outVal);
        out->addChild(input);

        out->setBackward([out, input, this]() {
            input->accumulateGrad(_backward(out->getData()) * out->getGrad());
        });

        return out;
    }

    /// @brief Element wise activation
    /// @param input Tensor
    /// @return output tensor 
    Tensor ElementWiseActivation::operator()(const Tensor& inputs) {
        Tensor outputs(inputs.dim());  // Create a tensor with the same dimensions as inputs

        for (int i = 0; i < inputs.size(); ++i) {
            outputs({i}) = (*this)(inputs({i}));  // Apply the forward operation on each element
        }

        return outputs;
    }

    Tensor TensorWiseActivation::operator()(const Tensor& inputs) {
        // Convert input Values to raw float data
        std::vector<float> x(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
            x[i] = inputs({i})->getData();
        }

        // Apply the forward function to compute the output
        std::vector<float> out = _forward_func(x);

        // Initialize outputs
        Tensor outputs(inputs.dim());
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
}