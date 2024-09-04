#pragma once
#include "../Tensor/Tensor.hpp"

namespace Losses
{
    ValRef meanSquareError(Tensor& pred, Tensor& real);
    ValRef meanAbsoluteError(Tensor& pred, Tensor& real);
    ValRef binaryCrossEntropy(Tensor& pred, Tensor& real);
} // namespace Losses
