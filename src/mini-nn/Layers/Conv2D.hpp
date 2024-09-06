#pragma once
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "Neuron.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Activations/Activation.hpp"

namespace Layers {
    class Conv2D : public Layer { // one dim layer
    private:
        std::vector<Neuron> neurons_;  // Neurons in the dense layer
        Tensor outputs_;  // Shared pointers to the outputs of the layer
        Activations::ActivationFunction func_;

    public:
        /// @brief Constructor
        Conv2D(int num_input_channel, int num_output_channel, int kernel_size = 3, int strides = 1);

        /// @brief Constructor
        Conv2D(int num_input_channel, int num_output_channel, int kernel_size = 3, int strides = 1, Activations::ActivationFunction func);

        /// @brief Static factory method
        static std::shared_ptr<Conv2D> create(int num_inputs, int num_outputs){
            return std::make_shared<Conv2D>(num_inputs, num_outputs);
        }

        /// @brief Static factory method
        static std::shared_ptr<Conv2D> create(int num_inputs, int num_outputs, Activations::ActivationFunction func){
            return std::make_shared<Conv2D>(num_inputs, num_outputs, func);
        }

        /// @brief Forward input into a neuron, should be called once at graph initialization
        // or only after computation graph was reset on every element.
        /// @param inputs inputs of dim (batch_size, num_inputs)
        /// @return output tensor, shape : (batch_size, num_outputs)
        const Tensor& forward(Tensor& inputs) override;


        /// @brief Get all parameters (weights) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor getParameters() override;

        /// @brief Get all parameters (biases) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor getBiases() override;

        std::vector<int> shape() { return outputs_.dim(); }
    };
}

