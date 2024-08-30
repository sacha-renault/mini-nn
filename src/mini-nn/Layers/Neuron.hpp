#pragma once
#include <vector>
#include <memory>
#include <random>
#include "../Values/Value.hpp" 
#include "../Tensor/Tensor.hpp" 
#include "../Operation/Math.hpp"

class Neuron {
private:
    Tensor wi_;  // Using shared_ptr for weights
    Tensor xiwi_;     // Using shared_ptr for xiwi output
    std::shared_ptr<Value> bias_;                  // Using shared_ptr for bias
    std::shared_ptr<Value> xnwn_;                  // result of the sum of all xiwi
    std::shared_ptr<Value> output_;                // Using shared_ptr for output

public:
    // Constructor
    Neuron(int num_inputs) : 
            xiwi_({ num_inputs }),
            output_(Value::create(0.0f))  {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Initialize weights
        wi_ = Tensor::randn({ num_inputs });

        // Create the bias point
        bias_ = Value::create(dist(gen));  // Initialize bias
    }

    // Forward pass
    std::shared_ptr<Value>& forward(const Tensor& xi) {
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
        xnwn_ = Math::reduceSum(xiwi_);

        // Add bias
        output_ = xnwn_->add(bias_);  // Set output
        return output_;
    }

    // Backward pass
    // void backward() {
    //     if (output_) {
    //         output_->backward();  // Backward pass for the output
    //     }
    // }

    // Getters
    Tensor& getWeights() { return wi_; }
    std::shared_ptr<Value>& getBias() { return bias_; }
    std::shared_ptr<Value>& getOutput() { return output_; }
};
