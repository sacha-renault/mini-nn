#pragma once
#include "../Tensor/Tensor.hpp"

namespace Losses
{
    std::shared_ptr<Value> meanSquareError(Tensor& pred, Tensor& real);
    std::shared_ptr<Value> meanAbsoluteError(Tensor& pred, Tensor& real);
    std::shared_ptr<Value> binaryCrossEntropy(Tensor& pred, Tensor& real);
} // namespace Losses
