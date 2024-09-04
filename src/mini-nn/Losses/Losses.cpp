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

    ValRef meanAbsoluteError(Tensor& pred, Tensor& real) {
        Tensor x = Math::ewSub(pred, real);
        x = Math::abs(x);
        ValRef loss = Math::reduceMean(x);
        return std::move(loss);
    }

    ValRef binaryCrossEntropy(Tensor& pred, Tensor& real) {
        if (pred.dim() != real.dim()) {
            throw std::invalid_argument("Predicted and real tensors must have the same dimensions for Binary Cross-Entropy loss.");
        }

        float epsilon = 1e-12;  // Small value to avoid log(0)
        ValRef loss = Value::create(0.0f);  // Initialize the loss value

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
} // namespace Losses
