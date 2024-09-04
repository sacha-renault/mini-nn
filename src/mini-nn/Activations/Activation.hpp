#pragma once
#include <functional>
#include "../Tensor/Tensor.hpp"

namespace Activations {
    // delegate typedef
    using ActivationFunction = std::function<Tensor(Tensor&)>;
    using ewActivationFunction = std::function<float(float)>;
    using twActivationFunction = std::function<float(std::vector<float>)>;

    Tensor _ewActivation(Tensor& input, ewActivationFunction forward, ewActivationFunction backward);

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