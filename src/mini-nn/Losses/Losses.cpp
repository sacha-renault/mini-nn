#include "Losses.hpp"
#include "../Operation/Math.hpp"
#include <iostream>

namespace Losses
{
    std::shared_ptr<Value> meanSquareError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::pow(x, 2);
        std::shared_ptr<Value> loss = Math::reduceMean(x);
        return std::move(loss);
    }

    std::shared_ptr<Value> meanAbsoluteError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::abs(x);
        std::shared_ptr<Value> loss = Math::reduceMean(x);
        return std::move(loss);
    }

    std::shared_ptr<Value> binaryCrossEntropy(Tensor& pred, Tensor& real) {
        if (pred.dim() != real.dim()) {
            throw std::invalid_argument("Predicted and real tensors must have the same dimensions for Binary Cross-Entropy loss.");
        }

        float epsilon = 1e-12;  // Small value to avoid log(0)
        std::shared_ptr<Value> loss = Value::create(0.0f);  // Initialize the loss value

        for (int i = 0; i < pred.size(); ++i) {
            auto p = pred.mat()[i];
            auto r = real.mat()[i];

            float p_val = std::min(std::max(p->getData(), epsilon), 1.0f - epsilon);  // Clamp the predicted value to avoid log(0)
            float r_val = r->getData();

            // Compute the binary cross-entropy for the current element
            float bce = -r_val * std::log(p_val) - (1.0f - r_val) * std::log(1.0f - p_val);

            // Accumulate the BCE into the loss
            loss->setValue(loss->getData() + bce);
            loss->addChild(p);
            loss->addChild(r);
            loss->addBackward([loss, p, r, p_val, r_val]() {
                float grad = loss->getGrad();
                float p_grad = grad * (-r_val / p_val + (1.0f - r_val) / (1.0f - p_val));
                p->accumulateGrad(p_grad);
            });

            // Set the forward function to recompute the BCE
            loss->addForward([loss, p, r, epsilon]() {
                float p_val = std::min(std::max(p->getData(), epsilon), 1.0f - epsilon);  // Clamp the predicted value to avoid log(0)
                float r_val = r->getData();
                float bce = -r_val * std::log(p_val) - (1.0f - r_val) * std::log(1.0f - p_val);
                loss->setValue(loss->getData() + bce);
            });
        }

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
            auto res = sum->times(neg);
            CEi({i}) = res;
        }
        return Math::reduceMean(CEi);
    }
} // namespace Losses
