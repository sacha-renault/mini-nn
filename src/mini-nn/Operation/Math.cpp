#include "Math.hpp"
#include "../Values/NodeOperator.hpp"
#include "../Values/NodeOperation.hpp"

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
        auto result = Value::create(total, std::make_unique<NodeOperation::Add>());

        // add other as childs
        for (const auto& val : tensor) {
            result->addChild(val);
        }

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
        auto result = Value::create(total / tensor.size(), std::make_unique<NodeOperation::Avg>());

        // add other as childs
        for (auto& val : tensor) {
            result->addChild(val);
        }
        return result;
    }


    std::shared_ptr<Value> pow(std::shared_ptr<Value> base, int exponent) {
        float base_value = base->getData();
        float pow_value = std::pow(base_value, exponent);

        std::shared_ptr<Value> result = Value::create(pow_value, std::make_unique<NodeOperation::Function1>(
            [exponent](float x) {
                return std::pow(x, exponent);
            },
            [exponent](float y) {
                float partialGradient = exponent * std::pow(y, exponent - 1);
                return partialGradient;
            }
        ));

        // Backward pass (for autograd)
        result->addChild(base);
        return result;
    }


    Tensor pow(Tensor& tensor, int exponent) {
        Tensor result(tensor.dim());
        for (int i = 0; i < tensor.size(); ++i) {
            result.mat()[i] = pow(tensor.mat()[i], exponent);
        }
        return result;
    }

    Tensor ewSum(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            result.mat()[i] = t1.mat()[i] + t2.mat()[i];
        }
        return result;
    }

    Tensor ewSub(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            result.mat()[i] = t1.mat()[i] - t2.mat()[i];
        }
        return result;
    }

    Tensor ewMul(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            result.mat()[i] = t1.mat()[i] * t2.mat()[i];
        }
        return result;
    }

    Tensor ewDiv(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            result.mat()[i] = t1.mat()[i] / t2.mat()[i];
        }
        return result;
    }

    Tensor abs(Tensor& t1) {
        Tensor result(t1.dim());

        // Iterate over each element in the tensor
        for (int i = 0; i < t1.size(); ++i) {
            auto val = t1.mat()[i];
            float absValue = std::fabs(val->getData());
            std::shared_ptr<Value> out = Value::create(absValue, std::make_unique<NodeOperation::Function1>(
                [](float x) {
                    return std::fabs(x);
                },
                [](float x) {
                    if (x >= 0) {
                        return 1.0f;
                    } else {
                        return -1.0f;
                    }

                }
            ));
            out->addChild(val);
            result.mat()[i] = out;
        }
        return result;
    }

    Tensor log(Tensor& t1) {
        Tensor result(t1.dim());

        // Iterate over each element in the tensor
        for (int i = 0; i < t1.size(); ++i) {
            auto val = t1.mat()[i];
            float logValue = std::log(val->getData());

            std::shared_ptr<Value> out = Value::create(logValue, std::make_unique<NodeOperation::Function1>(
                [](float x) {
                    return std::log(x);
                },
                [](float x) {
                    return 1.0f / x;
                }
            ));
            out->addChild(val);
            result.mat()[i] = out;
        }
        return result;
    }
} // namespace Math