#pragma once
#include "../Tensor/Tensor.hpp"

float sumExp(const Tensor& input) {
    float sum = 0.0f;
    for (const auto& val : input) {
        sum += std::exp(val->getData());
    }
    return sum;
}