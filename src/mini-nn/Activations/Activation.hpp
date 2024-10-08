#pragma once
#include <functional>
#include "../Tensor/Tensor.hpp"

namespace Activations {
    // delegate typedef
    using ActivationFunction = std::function<Tensor(Tensor&)>;

    Tensor _ewActivation(Tensor& input, std::function<float(float)> forward, std::function<float(float, float)> backward);

    /// @brief Activation ReLU
    /// @param input Tensor
    Tensor ReLU(Tensor& input);

    /// @brief Activation Tanh
    /// @param input Tensor
    Tensor Tanh(Tensor& input);

    /// @brief Activation Sigmoid
    /// @param input Tensor
    Tensor Sigmoid(Tensor& input);

    /// @brief Activation Softmax
    /// @param input Tensor
    Tensor Softmax(Tensor& input);
}