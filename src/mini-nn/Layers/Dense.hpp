#pragma once
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "Neuron.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Activations/ActivationWrapper.hpp"
#include "../Activations/ActivationFunction.hpp"

using namespace Activations;

namespace Layers {
    class Dense : public Layer { // one dim layer
    private:
        std::vector<Neuron> neurons_;  // Neurons in the dense layer
        Tensor outputs_;  // Shared pointers to the outputs of the layer
        ActivationWrapper func_;

    public:
        // Constructor
        Dense(int num_inputs, int num_outputs)
            : func_(), outputs_({num_outputs}) {
            for (int i = 0; i < num_outputs; ++i) {
                neurons_.emplace_back(Neuron(num_inputs));  // Initialize neurons with the number of inputs
            }
        }

        Dense(int num_inputs, int num_outputs, TensorWiseActivation func)
            :  Dense(num_inputs, num_outputs) {
                func_ = ActivationWrapper(std::make_shared<TensorWiseActivation>(func));
            }

        Dense(int num_inputs, int num_outputs, ElementWiseActivation func)
            :  Dense(num_inputs, num_outputs) {
                func_ = ActivationWrapper(std::make_shared<ElementWiseActivation>(func));
            }

        // Static factory method
        static std::shared_ptr<Dense> create(int num_inputs, int num_outputs){
            return std::make_shared<Dense>(num_inputs, num_outputs);
        }
        static std::shared_ptr<Dense> create(int num_inputs, int num_outputs, ElementWiseActivation func){
            return std::make_shared<Dense>(num_inputs, num_outputs, func);
        }

        // Forward pass for the dense layer
        const Tensor& forward(Tensor& inputs) override;

        /// @brief Backward pass for the dense layer
        // void backward() override;


        /// @brief Get all parameters (weights) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor getParameters() override;

        /// @brief Get all parameters (biases) from all neurons in the layer
        /// @return Tensor of parameters
        Tensor getBiases() override;

        std::vector<int> shape() { return outputs_.dim(); }
    };
}

