#include "Activation.hpp"

namespace Activations
{
    Tensor _ewActivation(Tensor& input, ewActivationFunction forward, ewActivationFunction backward) {
        Tensor output = Tensor::zeros(input.dim());             // init output tensor
        int i = 0;
        for (auto& val : input) {                               // Iterate over all input
            float x = val->getData();
            float y = forward(x);
            ValRef out = Value::create(y);                      // Create new node for graph
            out->addChild(val);                                 // Add child for new node

            out->setBackward([out, val, backward]() {
                val->accumulateGrad(backward(out->getData()) * out->getGrad());
            });

            out->setForward([out, val, forward]() {
                out->setValue(forward(val->getData()));
            });

            output({i++}) = out;                                            // New node added into the output
        }
        return std::move(output);
    };

    Tensor ReLU(Tensor& input) {
        ewActivationFunction f = [](float x) {
            return std::max(0.0f, x);
        };
        ewActivationFunction b = [](float x) {
            return x > 0.0f ? 1.0f : 0.0f;
        };
        return std::move(_ewActivation(input, f, b));
    }

    Tensor Tanh(Tensor& input) {
        ewActivationFunction f = [](float x) {
            return std::tanh(x);
        };
        ewActivationFunction b = [](float x) {
            return 1.0f - x * x;
        };
        return std::move(_ewActivation(input, f, b));
    }

    Tensor Sigmoid(Tensor& input) {
        ewActivationFunction f = [](float x) {
            return 1.0f / (1.0f + std::exp(-x));
        };
        ewActivationFunction b = [](float x) {
            return x * (1.0f - x);
        };
        return std::move(_ewActivation(input, f, b));
    }
} // namespace Activations
