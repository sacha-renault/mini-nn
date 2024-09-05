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
        result->addBackward([result, tensor]() {
            float upperNodeGradient = result->getGrad();
            for (auto& node : tensor) {
                node->accumulateGrad(upperNodeGradient);
            }
        });

        result->addForward([result, tensor]() {
            // Sum all the data from the other values
            float total = 0.0f;

            // Collect all the data and children references
            for (auto& val : tensor) {
                total += val->getData();
            }

            result->setValue(total);
        });
        return std::move(result);
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
        result->addBackward([result, tensor]() {
            float upperNodeGradient = result->getGrad();
            int size = tensor.size();
            for (auto& node : tensor) {
                node->accumulateGrad(upperNodeGradient / size);
            }
        });

        result->addForward([result, tensor]() {
            // Sum all the data from the other values
            float total = 0.0f;
            int size = tensor.size();
            // Collect all the data and children references
            for (auto& val : tensor) {
                total += val->getData();
            }

            result->setValue(total / size);
        });
        return std::move(result);
    }


    std::shared_ptr<Value> pow(std::shared_ptr<Value> base, int exponent) {
        float base_value = base->getData();
        float pow_value = std::pow(base_value, exponent);
        auto result = Value::create(pow_value);

        // Backward pass (for autograd)
        result->addChild(base);
        result->addBackward([base, base_value, exponent, result]() {
            float gradient = exponent * std::pow(base_value, exponent - 1);
            base->accumulateGrad(gradient * result->getGrad());
        });

        result->addForward([base, result, exponent]() {
            float floatResult = std::pow(base->getData(), exponent);
            result->setValue(floatResult);
        });

        return std::move(result);
    }


    Tensor pow(Tensor& tensor, int exponent) {
        Tensor result(tensor.dim());
        for (int i = 0; i < tensor.size(); ++i) {
            result.mat()[i] = pow(tensor.mat()[i], exponent);
        }
        return std::move(result);
    }

    Tensor pow(Tensor& t1, Tensor& t2) {
        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            result.mat()[i] = pow(t1.mat()[i], t2.mat()[i]->getData());
        }
        return std::move(result);
    }

    Tensor ewSum(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            auto t1_elt = t1.mat()[i];
            auto t2_elt = t2.mat()[i];
            result.mat()[i] = t1_elt->add(t2_elt);
        }
        return std::move(result);
    }

    Tensor ewSub(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            auto t1_elt = t1.mat()[i];
            auto t2_elt = t2.mat()[i];
            auto res = t1_elt->sub(t2_elt);
            result.mat()[i] = res;
        }
        return std::move(result);
    }

    Tensor ewMul(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            auto t1_elt = t1.mat()[i];
            auto t2_elt = t2.mat()[i];
            result.mat()[i] = t1_elt->times(t2_elt);
        }
        return std::move(result);
    }

    Tensor ewDiv(Tensor& t1, Tensor& t2) {
        if (t1.size() != t2.size()) { // no need for equal dim, just need equal size
            throw std::invalid_argument("Tensor dimensions must match for element-wise sum.");
        }

        Tensor result(t1.dim());
        for (int i = 0; i < t1.size(); ++i) {
            auto t1_elt = t1.mat()[i];
            auto t2_elt = t2.mat()[i];
            result.mat()[i] = t1_elt->div(t2_elt);
        }
        return std::move(result);
    }

    Tensor abs(Tensor& t1) {
        Tensor result(t1.dim());

        // Iterate over each element in the tensor
        for (int i = 0; i < t1.size(); ++i) {
            auto val = t1.mat()[i];
            float absValue = std::fabs(val->getData());
            ValRef out = Value::create(absValue);
            out->addChild(val);

            out->addBackward([out, val]() {
                float gradient = out->getGrad();
                float gradInput = (val->getData() >= 0) ? gradient : -gradient;
                val->accumulateGrad(gradInput);
            });

            out->addForward([out, val]() {
                float newValue = std::fabs(val->getData());
                out->setValue(newValue);
            });
            result.mat()[i] = out;
        }
        return std::move(result);
    }

    Tensor cloneWithGraph(Tensor& t1) {
        Tensor result(t1.dim());
        int size = t1.size();
        for(int i = 0 ; i < size ; ++i) {
            auto val = t1.mat()[i];
            auto out = Value::create(val->getData());
            out->addChild(val);
            out->addForward([val, out]() {
                out->setValue(val->getData());
            });
            out->addBackward([val, out]() {
                val->accumulateGrad(out->getGrad());
            });
            result.mat()[i] = out;
        }
        return std::move(result);
    }
} // namespace Math