#pragma once
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "Neuron.hpp"
#include "../Values/Tensor.hpp"
#include "../Activations/ActivationWrapper.hpp"
#include "../Activations/ActivationFunction.hpp"

using namespace Activations;

namespace Layers {
    class Dense : public Layer<1> { // one dim layer
    private:
        std::vector<Neuron> neurons_;  // Neurons in the dense layer
        Tensor<1> outputs_;  // Shared pointers to the outputs of the layer
        ActivationWrapper func_;

    public:
        // Constructor
        Dense(int num_inputs, int num_outputs) : func_() {
            for (int i = 0; i < num_outputs; ++i) {
                neurons_.emplace_back(Neuron(num_inputs));  // Initialize neurons with the number of inputs
            }
            outputs_.resize({num_outputs});  // Pre-allocate space for outputs
        }

        Dense(int num_inputs, int num_outputs, TensorWiseActivation func) 
            :  Dense(num_inputs, num_outputs) {
                func_ = ActivationWrapper(std::make_shared<TensorWiseActivation>(func));
            }
        
        Dense(int num_inputs, int num_outputs, ElementWiseActivation func) 
            :  Dense(num_inputs, num_outputs) {
                func_ = ActivationWrapper(std::make_shared<ElementWiseActivation>(func));
            }

        // Forward pass for the dense layer
        Tensor<1>& forward(Tensor<1>& inputs) override;

        /// @brief Backward pass for the dense layer
        void backward() override;

        
        /// @brief Get all parameters (weights) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor<1> getParameters() override;

        /// @brief Get all parameters (biases) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor<1> getBiases() override;
    };
}

