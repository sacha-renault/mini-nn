#include "Activation.hpp"
#include "../Utils/HelperFunc.hpp"

namespace Activations
{
    Tensor _ewActivation(Tensor& input, ewActivationFunction forward, ewActivationFunction backward) {
        Tensor output = Tensor({input.size()});                   // init output tensor
        int i = 0;
        for (auto& val : input) {                               // Iterate over all input
            float x = val->getData();
            float y = forward(x);
            ValRef out = Value::create(y);                      // Create new node for graph
            out->addChild(val);                                 // Add child for new node

            out->addBackward([out, val, backward]() {
                val->accumulateGrad(backward(out->getData()) * out->getGrad());
            });

            out->addForward([out, val, forward]() {
                out->setValue(forward(val->getData()));
            });

            output({i++}) = out;                                            // New node added into the output
        }
        output.reshape(input.dim());
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

    Tensor Softmax(Tensor& input) {
        if (input.rank() != 2) {
            throw std::runtime_error("Softmax require a tensor of rank 2. (batchSize, featureSize)");
        }
        Tensor output(input.dim());
        int batchSize = input.dim()[0];  // Assume input is (batchSize, features)
        int featureSize = input.dim()[1];

        for (int i = 0; i < batchSize; ++i) {
            // Extract the current slice (1D tensor for this batch entry)
            Tensor inputSlice = input[i];

            // Compute the sum of exponentials
            float expSum = sumExp(inputSlice);

            // Iterate through the slice to apply the Softmax function
            for (int j = 0; j < featureSize; ++j) {
                auto val = inputSlice({j});
                float softmaxValue = std::exp(val->getData()) / expSum;

                ValRef out = Value::create(softmaxValue);  // Create a new node for the graph
                out->addChild(val);  // Add the current value as a child

                out->addBackward([out, val, softmaxValue]() {
                    float gradient = out->getGrad();
                    float gradInput = softmaxValue * (gradient - softmaxValue * gradient);
                    val->accumulateGrad(gradInput);
                });

                out->addForward([out, val, expSum]() {
                    float softmaxValue = std::exp(val->getData()) / expSum;
                    out->setValue(softmaxValue);
                });

                output({i, j}) = out;  // Store the result in the output tensor
            }
        }

        return std::move(output);
    }
} // namespace Activations
