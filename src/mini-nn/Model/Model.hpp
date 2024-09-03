#pragma once
#include "../Tensor/Tensor.hpp"
#include "../Values/Value.hpp"

class Model{
public:
    virtual const Tensor& forward(Tensor& input) = 0;
    virtual std::vector<std::shared_ptr<Value>> getParameters() = 0;
    virtual void update(float lr) = 0;
};