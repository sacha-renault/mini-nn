#pragma once
#include <vector>
#include "../Values/Value.hpp"
#include "../Values/Parameter.hpp"
#include "../Tensor/Tensor.hpp"

class Layer {
public:
    // Virtual destructor for the base class
    virtual ~Layer() = default;

    // Pure virtual method for the forward pass
    virtual Tensor& forward(Tensor& inputs) = 0;

    // Pure virtual method for the backward pass (if needed)
    virtual void backward() = 0;

    // Pure virtual method to get the parameters of the layer
    virtual Tensor getParameters() = 0;

    // Any additional virtual methods can be added here as needed
    virtual Tensor getBiases() = 0;
};