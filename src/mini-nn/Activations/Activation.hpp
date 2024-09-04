#pragma once
#include <functional>
#include "../Tensor/Tensor.hpp"

namespace Activations {
    // delegate typedef
    using ActivationFunction = std::function<Tensor(Tensor&)>;
    using ewActivationFunction = std::function<float(float)>;
    using twActivationFunction = std::function<float(std::vector<float>)>;

    Tensor _ewActivation(Tensor& input, ewActivationFunction forward, ewActivationFunction backward);

    // ReLU
    Tensor ReLU(Tensor& input);
    Tensor Tanh(Tensor& input);
    Tensor Sigmoid(Tensor& input);

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