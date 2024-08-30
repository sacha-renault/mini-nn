#pragma once
#include <vector>
#include <memory>
#include <random>
#include "../Values/Value.hpp"  // Ensure you include the updated Value class

class Neuron {
private:
    std::vector<std::shared_ptr<Value>> wi_;  // Using shared_ptr for weights
    std::vector<std::shared_ptr<Value>> xiwi_;     // Using shared_ptr for xiwi output
    std::shared_ptr<Value> bias_;                  // Using shared_ptr for bias
    std::shared_ptr<Value> output_;                // Using shared_ptr for output

public:
    // Constructor
    Neuron(size_t num_inputs) : output_(Value::create(0.0f)) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < num_inputs; ++i) {
            wi_.emplace_back(Value::create(dist(gen)));  // Initialize weights
        }

        // Resize xiwi_ to match the number of inputs
        xiwi_.resize(num_inputs);

        // Create the bias point
        bias_ = Value::create(dist(gen));  // Initialize bias
    }

    // Forward pass
    std::shared_ptr<Value>& forward(const std::vector<std::shared_ptr<Value>>& xi) {
        if (xi.size() != wi_.size()) {
            throw std::invalid_argument("Input size must match the number of weights.");
        }


        // Compute the weighted inputs
        for (size_t i = 0; i < xi.size(); ++i) {
            xiwi_[i] = xi[i]->times(wi_[i]);
        }

        // Sum all weighted inputs and add bias
        auto weighted_sum = sumManyValue(xiwi_);

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
    std::vector<std::shared_ptr<Value>>& getWeights() { return wi_; }
    std::shared_ptr<Value>& getBias() { return bias_; }
    std::shared_ptr<Value>& getOutput() { return output_; }
};
