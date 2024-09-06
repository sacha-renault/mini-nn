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
} // namespace Optimizers


