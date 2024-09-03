#pragma once
#include "../Tensor/Tensor.hpp"

namespace Losses
{
    ValRef meanSquareError(Tensor& pred, Tensor& real);
} // namespace Losses
