#pragma once
#include <vector>
#include <functional>
#include <random>
#include "Parameter.hpp"
#include "ActivationFunction.hpp"

class Neuron {
private:
    // Weights and bias
    std::vector<Parameter> weights_;  // Parameters (weights) to learn
    Parameter bias_;                  // Parameter (bias) to learn

    // Intermediate values (needed to keep track for backpropagation)
    std::vector<Value> xiwiOutput_;   // Stores the output from each individual multiplication
    Value xnwnOutput_;                // Sum of all xiwi
    Value biasOutput_;                // Final output after adding bias

public:
    // Constructor
    Neuron(size_t num_inputs)
        : bias_(0.0f), xnwnOutput_(0.0f), biasOutput_(0.0f) {

        // Initialize weights and bias
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < num_inputs; ++i) {
            weights_.emplace_back(Value(dist(gen)));    // Random weight init
            xiwiOutput_.emplace_back(Value(0));         // Dummy init
        }
        bias_ = Parameter(dist(gen));
    }

    // Forward pass for the neuron
    Value forward(std::vector<Value>& inputs) {
        if (inputs.size() != weights_.size()) {
            throw std::invalid_argument("Input size must match the number of weights.");
        }

        // Weighted sum of inputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            xiwiOutput_[i] = (weights_[i] * inputs[i]);
        }

        // get pointers
        std::vector<Value*> xiwiOutputPtrs;
        for (auto& xiwi : xiwiOutput_) {
            xiwiOutputPtrs.push_back(&xiwi);
        }
        
        xnwnOutput_ = sumManyValue(xiwiOutputPtrs);
        biasOutput_ = xnwnOutput_ + bias_;
        return biasOutput_; // Return the first (and only) element
    }

    // Getters for weights and bias
    std::vector<Parameter>& getWeights() { return weights_; }
    Parameter& getBias() { return bias_; }
};