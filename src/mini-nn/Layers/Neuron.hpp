#pragma once
#include <vector>
#include <memory>
#include <random>
#include "../Values/Value.hpp"
#include "../Values/NodesTypes.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Operation/Math.hpp"

class Neuron {
private:
    Tensor wi_;                                     // Using shared_ptr for weights
    std::shared_ptr<Value> bias_;                   // Using shared_ptr for bias
    Tensor output_;                                 // Using shared_ptr for output

public:
    /// @brief Neuron construct
    /// @param num_inputs number of input that will be linked to a single output
    Neuron(int num_inputs);

    /// @brief Forward input into a neuron, should be called once at graph initialization
    // or only after computation graph was reset on every element.
    /// @param xi input of dim (batch_size, num_inputs)
    /// @return output tensor, shape : (batch_size, )
    Tensor& forward(Tensor& xi);

    /// @brief Get the wi_ weights
    /// @return tensor representing the weights of the neuron
    Tensor& getWeights() { return wi_; }

    /// @brief Get the bias param of the neuron
    /// @return bias
    std::shared_ptr<Value>& getBias() { return bias_; }

    /// @brief get the output of the last graph execution
    /// @return output tensor of shape (batch_size, )
    Tensor& getOutput() { return output_; }
};
