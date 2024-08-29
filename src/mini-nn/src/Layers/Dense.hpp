#pragma once
#include <vector>
#include <memory>
#include "Layer.hpp"
#include "Neuron.hpp"

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

    Activations::LambdaActivation *func_;

    // Forward pass for the dense layer
    std::vector<std::shared_ptr<Value>>& forward(std::vector<std::shared_ptr<Value>>& inputs) override {
        for (size_t i = 0; i < neurons_.size(); ++i) {
            outputs_[i] = neurons_[i].forward(inputs);  // Store each neuron's output
        }

        if (func_) {
            outputs_ = (*func_)(outputs_);
        }

        return outputs_;
    }

    // Backward pass for the dense layer
    void backward() override {
        for (auto& out : outputs_) {
            out->backward();  // Perform backpropagation for each neuron's output
        }
    }

    // Get all parameters (weights and biases) from all neurons in the layer
    std::vector<std::shared_ptr<Value>> getParameters() override {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& neuron : neurons_) {
            auto weights = neuron.getWeights();  // Get weights of each neuron
            params.insert(params.end(), weights.begin(), weights.end());
        }
        return params;
    }

    std::vector<std::shared_ptr<Value>> getBiases() {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& neuron : neurons_) {
            auto weights = neuron.getWeights();  // Get weights of each neuron
            params.push_back(neuron.getBias());
        }
        return params;
    }
};
