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
    /// @param input Tensor1D
    /// @return output tensor 
    Tensor1D ElementWiseActivation::operator()(const Tensor1D& inputs) {
        Tensor1D outputs(inputs.dimensions());  // Create a tensor with the same dimensions as inputs

        for (int i = 0; i < inputs.size(); ++i) {
            outputs({i}) = (*this)(inputs({i}));  // Apply the forward operation on each element
        }

        return outputs;
    }

    /// @brief Element wise activation
    /// @param input Tensor
    /// @return output tensor 
    template<typename TensorType>
    TensorType ElementWiseActivation::operator()(const TensorType& inputs) {
        // Save the original shape
        auto original_shape = inputs.dimensions();

        // Prepare a new shape for (n, 1, 1, ...)
        std::array<Eigen::Index, TensorType::Rank> reshaped_dims;
        reshaped_dims.fill(1);
        reshaped_dims[0] = inputs.size();  // Set the first dimension to n

        // Reshape the inputs to (n, 1, 1, ...)
        TensorType reshaped_inputs = inputs;
        reshaped_inputs.resize(reshaped_dims);

        // Initialize the output tensor with the reshaped dimensions
        TensorType reshaped_outputs(reshaped_dims);

        // Loop over the first dimension (n)
        for (int i = 0; i < reshaped_inputs.size(); ++i) {
            std::array<Eigen::Index, TensorType::Rank> index = {};
            index.fill(1);
            index[0] = i;

            reshaped_outputs(index) = (*this)(reshaped_inputs(index));
        }

        // Reshape the output tensor back to the original shape
        reshaped_outputs.resize(original_shape);
        return reshaped_outputs;
    }

    Tensor1D Tensor1DWiseActivation::operator()(const Tensor1D& inputs) {
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
}