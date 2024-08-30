#include "Math.hpp"

namespace Math
{
    std::shared_ptr<Value> reduceSum(Tensor& tensor) {
        // Sum all the data from the other values
        float total = 0.0f;

        // Collect all the data and children references
        for (auto val : tensor) {
            total += val->getData();
        }

        // Create a new Value object for the sum with the collected children
        auto result = Value::create(total);

        // add other as childs
        for (auto val : tensor) {
            result->addChild(val);
        }

        // Set the backward function to distribute the gradient to all children
        result->setBackward([result, tensor]() {
            float upperNodeGradient = result->getGrad();
            for (auto child : tensor) {
                if (child) {
                    child->accumulateGrad(upperNodeGradient);
                }
            }
        });

        return result;
    }
} // namespace Math