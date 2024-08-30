#pragma once
#include <vector>
#include <memory>
#include <random>
#include "../Values/Value.hpp" 
#include "../Values/Tensor.hpp" 

class Neuron {
private:
    Tensor<1> wi_;  // Using shared_ptr for weights
    Tensor<1> xiwi_;     // Using shared_ptr for xiwi output
    std::shared_ptr<Value> bias_;                  // Using shared_ptr for bias
    std::shared_ptr<Value> output_;                // Using shared_ptr for output

public:
    // Constructor
    Neuron(int num_inputs) : 
            wi_({num_inputs}), 
            xiwi_({num_inputs}),
            output_(Value::create(0.0f))  {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Initialize weights
        for (int i = 0; i < num_inputs; ++i) {
            wi_({i}) = Value::create(dist(gen));
        }

        // Resize xiwi_ to match the number of inputs
        xiwi_.resize({num_inputs});

        // Create the bias point
        bias_ = Value::create(dist(gen));  // Initialize bias
    }

    // Forward pass
    std::shared_ptr<Value>& forward(const Tensor<1>& xi) {
        // First assert that Tensor is rank one
        if (xi.rank() != 1) {
            throw std::invalid_argument("Input dim must be 1 for Neuron.");
        }

        // Assert dim are equal
        if (xi.dim() != wi_.dim()) {
            throw std::invalid_argument("Input size must match the number of weights.");
        }


        // Compute the weighted inputs  by iterating over axis 0;
        for (int i = 0; i < xi.dim()[0]; ++i) {
            xiwi_({i}) = xi({i})->times(wi_({i}));
        }

        // Sum all weighted inputs and add bias
        auto weighted_sum = xiwi_.sum();

        // Add bias
        output_ = weighted_sum->add(bias_);  // Set output
        return output_;
    }

    // Backward pass
    // void backward() {
    //     if (output_) {
    //         output_->backward();  // Backward pass for the output
    //     }
    // }

    // Getters
    Tensor<1>& getWeights() { return wi_; }
    std::shared_ptr<Value>& getBias() { return bias_; }
    std::shared_ptr<Value>& getOutput() { return output_; }
};
