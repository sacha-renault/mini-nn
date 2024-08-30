#pragma once
#include <memory>
#include "../Tensor/Tensor.hpp"

namespace Math
{
    inline std::shared_ptr<Value> reduceSum(Tensor& others) {
        // Sum all the data from the other values
        float total = 0.0f;

        // Collect all the data and children references
        for (auto val : others) {
            total += val->getData();
        }

        // Create a new Value object for the sum with the collected children
        auto result = Value::create(total);

        // add other as childs
        for (auto val : others) {
            result->addChild(val);
        }

        // Set the backward function to distribute the gradient to all children
        result->setBackward([result, others]() {
            float upperNodeGradient = result->getGrad();
            for (auto child : others) {
                if (child) {
                    child->accumulateGrad(upperNodeGradient);
                }
            }
        });

        return result;
    }
} // namespace Math
