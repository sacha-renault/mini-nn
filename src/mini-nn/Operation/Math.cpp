#include "Math.hpp"

namespace Math
{
    std::shared_ptr<Value> reduceSum(Tensor& tensor) {
        // Sum all the data from the other values
        float total = 0.0f;

        // Collect all the data and children references
        for (const auto& val : tensor) {
            total += val->getData();
        }

        // Create a new Value object for the sum with the collected children
        auto result = Value::create(total);

        // add other as childs
        for (const auto& val : tensor) {
            result->addChild(val);
        }

        // Set the backward function to distribute the gradient to all children
        result->setBackward([result, tensor]() {
            float upperNodeGradient = result->getGrad();
            for (auto& node : tensor) {
                node->accumulateGrad(upperNodeGradient);
            }
        });

        result->setForward([result, tensor]() {
            // Sum all the data from the other values
            float total = 0.0f;

            // Collect all the data and children references
            for (auto& val : tensor) {
                total += val->getData();
            }

            result->setValue(total);
        });
        return result;
    }


    std::shared_ptr<Value> reduceMean(Tensor& tensor) {
        // Sum all the data from the other values
        float total = 0.0f;

        // Collect all the data and children references
        for (auto& val : tensor) {
            total += val->getData();
        }

        // Create a new Value object for the sum with the collected children
        auto result = Value::create(total / tensor.size());

        // add other as childs
        for (auto& val : tensor) {
            result->addChild(val);
        }

        // Set the backward function to distribute the gradient to all children
        result->setBackward([result, tensor]() {
            float upperNodeGradient = result->getGrad();
            for (auto& node : tensor) {
                node->accumulateGrad(upperNodeGradient);
            }
        });

        tensor.mat()[0]->setForward([result, tensor]() {
            // Sum all the data from the other values
            float total = 0.0f;

            // Collect all the data and children references
            for (auto& val : tensor) {
                total += val->getData();
            }

            result->setValue(total / tensor.size());
        });
        return result;
    }


    std::shared_ptr<Value> pow(std::shared_ptr<Value> base, int exponent) {
        float base_value = base->getData();
        float pow_value = std::pow(base_value, exponent);

        auto result = Value::create(pow_value);

        // Backward pass (for autograd)
        result->addChild(base);
        result->setBackward([base, base_value, exponent, result]() {
            float gradient = exponent * std::pow(base_value, exponent - 1);
            base->accumulateGrad(gradient * result->getGrad());
        });

        base->setForward([base, result, exponent]() {
            float floatResult = std::pow(base->getData(), exponent);
            result->setValue(floatResult);
        });

        return std::move(result);
    }


    Tensor pow(Tensor& tensor, int exponent) {
        Tensor result(tensor.dim());
        for (int i = 0; i < tensor.size(); ++i) {
            result({i}) = pow(tensor({i}), exponent);
        }
        return result;
    }
} // namespace Math