#pragma once
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <vector>
#include "../Values/Value.hpp"
#include "../Values/Tensor.hpp"

namespace Activations {

    /// @brief Base activation class
    class BaseActivation { 
    public:
        virtual Tensor1D operator()(const Tensor1D& inputs) = 0;
    };



    /// @brief ElementWiseActivation, each output depend only of one input
    class ElementWiseActivation : public BaseActivation {
    protected:
        std::function<float(float)> _forward_func;
        std::function<float(float)> _backward_func;

        float _forward(float x) {
            return _forward_func(x);
        }

        float _backward(float x) {
            return _backward_func(x);
        } 

    public:
        ElementWiseActivation(std::function<float(float)> forward, std::function<float(float)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        virtual std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& input) final;
        virtual Tensor1D operator()(const Tensor1D& inputs) override;
        template<typename TensorType>
        TensorType operator()(const TensorType& inputs);
    };



    /// @brief Tensor1DWiseActivation, each output depend all of the inputs
    class Tensor1DWiseActivation : public BaseActivation {
    protected:
        std::function<std::vector<float>(const std::vector<float>& x)> _forward_func;
        std::function<std::vector<float>(const std::vector<float>& x)> _backward_func;

        std::vector<float> _forward(const std::vector<float>& x) {
            return _forward_func(x);
        }

        std::vector<float> _backward(const std::vector<float>& x) {
            return _backward_func(x);
        }

    public:
        Tensor1DWiseActivation(
            std::function<std::vector<float>(const std::vector<float>& x)> forward,
            std::function<std::vector<float>(const std::vector<float>& x)> backward)
            : _forward_func(forward), _backward_func(backward) { }

        virtual Tensor1D operator()(const Tensor1D& inputs) override;
    };
}
