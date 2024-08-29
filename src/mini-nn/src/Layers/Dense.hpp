#pragma once
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "Neuron.hpp"
#include "../Activations/Static.hpp"

namespace Layers {
    class Dense : public Layer {
    private:
        std::vector<Neuron> neurons_;  // Neurons in the dense layer
        std::vector<std::shared_ptr<Value>> outputs_;  // Shared pointers to the outputs of the layer
        

    public:
        // Constructor
        Dense(size_t num_inputs, size_t num_outputs) {
            for (size_t i = 0; i < num_outputs; ++i) {
                neurons_.emplace_back(Neuron(num_inputs));  // Initialize neurons with the number of inputs
            }
            outputs_.resize(num_outputs);  // Pre-allocate space for outputs
        }

        Activations::LambdaActivation *func_ = nullptr;

        // Forward pass for the dense layer
        std::vector<std::shared_ptr<Value>>& forward(std::vector<std::shared_ptr<Value>>& inputs) override;

        /// @brief Backward pass for the dense layer
        void backward() override;

        
        /// @brief Get all parameters (weights) from all neurons in the layer
        /// @return Tensor of parameters
        std::vector<std::shared_ptr<Value>> getParameters();

        /// @brief Get all parameters (biases) from all neurons in the layer
        /// @return Tensor of parameters
        std::vector<std::shared_ptr<Value>> getBiases();
    };
}

