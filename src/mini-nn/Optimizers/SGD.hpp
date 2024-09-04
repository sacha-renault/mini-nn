#pragma once
#include "Optimizer.hpp"

namespace Optimizers
{
    class SGD : public Optimizer
    {
    private:
        std::vector<ValRef> params;

    public:
        SGD(Model& model, float lr);
        virtual void update(ValRef &loss) override;
    };

    SGD::SGD(Model& model, float lr)
    {
        lr_ = lr;
        params = model.getParameters();
    }

    void SGD::update(ValRef &loss)
    {
        auto grad = Gradient::reverseTopologicalOrder(loss);
        Gradient::backward(grad);
        // Gradient::noiseGrad(grad, 1e4);
        Gradient::clipGrad(grad, 1.5f);
        for(auto& node : params) {
            float grad = node->getGrad();
            float data = node->getData();
            node->setValue(data - grad * lr_);
        }
        Gradient::zeroGrad(grad);
    }
} // namespace Optimizers


