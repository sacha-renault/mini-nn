#include "Adam.hpp"

namespace Optimizers
{
    Adam::Adam(Model& model, float lr, float beta1, float beta2, float epsilon)
        : beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(1)
    {
        lr_ = lr; // from optimizer
        for(auto& val : model.getParameters()) {
            params_.push_back(ParamMomentumState(val));
        }
    }

    void Adam::update(ValRef &loss)
    {
        auto grad = Gradient::reverseTopologicalOrder(loss);
        Gradient::backward(grad);
        // Gradient::noiseGrad(grad, 1e4);
        Gradient::clipGrad(grad, 1.5f);
        for(auto& node : params_) {
            updateNode(node);
        }
        Gradient::zeroGrad(grad);
        ++t_;
    }

    void Adam::updateNode(ParamMomentumState& container) {
        float grad = container.node->getGrad();
        if (container.node->getType() == NodeTypes::BIAS) {
            container.m = beta1_ * container.m + (1 - beta1_) * grad;
            container.s = beta2_ * container.s + (1 - beta2_) * std::pow(grad, 2);
        } else if (container.node->getType() == NodeTypes::WEIGHT) {
            container.m = beta1_ * container.m + (1 - beta1_) * grad;
            container.s = beta2_ * container.s + (1 - beta2_) * std::pow(grad, 2);
        } else {
            throw std::runtime_error("Unknown type");
        }

        // Calculate corrector
        float mc = container.m / (1 - std::pow(beta1_, t_));
        float ms = container.s / (1 - std::pow(beta2_, t_));

        // Update weight and biases
        float currentValue = container.node->getData();
        float newValue = currentValue - lr_ * mc / (std::sqrt(ms) + epsilon_);
        container.node->setValue(newValue);
    }
} // namespace Optimizers
