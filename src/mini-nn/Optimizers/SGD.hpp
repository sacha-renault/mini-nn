#pragma once
#include "Optimizer.hpp"

namespace Optimizers
{
    class SGD : public Optimizer
    {
    private:
        std::vector<std::shared_ptr<Value>> params;

    public:
        SGD(Model& model, float lr);
        virtual void update(std::shared_ptr<Value> &loss) override;
    };
} // namespace Optimizers


