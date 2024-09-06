#pragma once
#include "Optimizer.hpp"


namespace Optimizers // namespace name
{
    struct ParamMomentumState
    {
        std::shared_ptr<Value> node;
        float m; // 1st moment
        float s; // second moment
        ParamMomentumState(std::shared_ptr<Value>& n) : node(n), m(0), s(0) { }
    };


    class Adam : public Optimizer
    {
    private:
        std::vector<ParamMomentumState> params_;
        float beta1_;
        float beta2_;
        float epsilon_;
        int t_;

        void updateNode(ParamMomentumState& node);
    public:
        Adam(Model& model, float lr = 0.01, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
        virtual void update(std::shared_ptr<Value> &loss) override;
    };
}