#pragma once
#include "../Values/Value.hpp"
#include "../Model/Model.hpp"
#include "../Operation/Gradient.hpp"

class Optimizer
{
protected:
    float lr_;

public:
    virtual void update(ValRef &loss) = 0;
    virtual void setLearningRate(float lr) { lr_ = lr; }
    virtual float getLearningRate() { return lr_; }
};
