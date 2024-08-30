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


    Tensor pow(Tensor& tensor, int exponent) {
        Tensor result(tensor.dim());

        for (int i = 0; i < tensor.size(); ++i) {
            float base_value = tensor({i})->getData();
            float pow_value = std::pow(base_value, exponent);

            auto new_value = std::make_shared<Value>(pow_value);

            // Backward pass (for autograd)
            new_value->addChild(tensor({i}));
            new_value->setBackward([tensor_data = tensor({i}), base_value, exponent]() {
                float gradient = exponent * std::pow(base_value, exponent - 1);
                tensor_data->accumulateGrad(gradient * tensor_data->getGrad());
            });
            result({i}) = new_value;
        }

        return result;
    }
} // namespace Math