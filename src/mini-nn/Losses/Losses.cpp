#include "Losses.hpp"
#include "../Operation/Math.hpp"

namespace Losses
{
    ValRef meanSquareError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::pow(x, 2);
        ValRef loss = Math::reduceMean(x);
        return std::move(loss);
    }
} // namespace Losses
