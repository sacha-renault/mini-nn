#include "Losses.hpp"
#include "../Values/NodeOperator.hpp"
#include "../Operation/Math.hpp"

namespace Losses
{
    std::shared_ptr<Value> meanSquareError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::pow(x, 2);
        auto loss = Math::reduceMean(x);
        return loss;
    }

    std::shared_ptr<Value> meanAbsoluteError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::abs(x);
        auto loss = Math::reduceMean(x);
        return loss;
    }

    std::shared_ptr<Value> categoricalCrossEntropy(Tensor& pred, Tensor& real) {
        if (pred.dim() != real.dim() || real.rank() != 2) {
            throw std::runtime_error("Dim of pred and real must be equal, and rank must be 2");
        }
        Tensor CEi({pred.dim()[0]});
        auto neg = Value::create(-1.0f);
        for (int i = 0 ; i < real.dim()[0] ; ++i) { // for each batch
            auto log = Math::log(pred);
            auto prod = Math::ewMul(real, log);
            auto sum = Math::reduceSum(prod);
            auto res = sum * neg;
            CEi({i}) = res;
        }
        return Math::reduceMean(CEi);
    }
} // namespace Losses
