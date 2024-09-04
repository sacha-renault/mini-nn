#pragma once
#include "../Values/Value.hpp"
#include "../Model/Model.hpp"

class Optimizer
{
public:
    virtual void update(ValRef &loss) = 0;
    virtual void setLearningRate(float lr) = 0;
    virtual float getLearningRate() = 0;
};
