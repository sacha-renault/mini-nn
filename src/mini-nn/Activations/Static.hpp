#pragma once
#include "ActivationFunction.hpp"

namespace Activations {
    // ReLU activation function
    static ElementWiseActivation ReLU(
        [](float x) { return std::max(0.0f, x); },           // Forward: ReLU
        [](float x) { return x > 0.0f ? 1.0f : 0.0f; }       // Backward: ReLU derivative
    );

    // Tanh activation function
    static ElementWiseActivation Tanh(
        [](float x) { return std::tanh(x); },                // Forward: Tanh
        [](float x) { return 1.0f - x * x; }  // Backward: Tanh derivative
    );

    // Sigmoid activation function
    static ElementWiseActivation Sigmoid(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },  // Forward: Sigmoid
        [](float x) { return x * (1.0f - x); }  // Backward: Sigmoid derivative
    );

    // Softmax activation function
    // static TensorWiseActivation Softmax(
    //     // Forward function
    //     [](const std::vector<float>& x) -> std::vector<float> {
    //         std::vector<float> exp_x(x.size());
    //         float max_val = *std::max_element(x.begin(), x.end());
            
    //         // Compute exponentials for numerical stability
    //         for (size_t i = 0; i < x.size(); ++i) {
    //             exp_x[i] = std::exp(x[i] - max_val);
    //         }
            
    //         float sum_exp_x = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f);

    //         // Normalize to get probabilities
    //         for (size_t i = 0; i < exp_x.size(); ++i) {
    //             exp_x[i] /= sum_exp_x;
    //         }

    //         return exp_x;
    //     },
        
    //     // Backward function
    //     [](const std::vector<float>& softmax_output) -> std::vector<float> {
    //         std::vector<float> grad(softmax_output.size());
    //         // Compute the gradient for each output using the simplified diagonal Jacobian form
    //         for (size_t i = 0; i < softmax_output.size(); ++i) {
    //             grad[i] = softmax_output[i] * (1.0f - softmax_output[i]);
    //         }
    //         return grad;
    //     }
    // );
}