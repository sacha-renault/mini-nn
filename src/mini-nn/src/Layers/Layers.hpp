#pragma once
#include <vector>
#include "../Value.hpp"
#include "../Parameter.hpp"

class Layer {
public:
    // Virtual destructor for the base class
    virtual ~Layer() = default;

    // Pure virtual method for the forward pass
    virtual std::vector<Value> forward(const std::vector<Value>& inputs) = 0;

    // Pure virtual method for the backward pass (if needed)
    virtual void backward() = 0;

    // Pure virtual method to get the parameters of the layer
    virtual std::vector<Parameter*> getParameters() = 0;

    // Any additional virtual methods can be added here as needed
};