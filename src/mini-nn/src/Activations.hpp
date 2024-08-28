#pragma once
#include <cmath>
#include "Value.hpp"
#include "ActivationFunction.hpp"

namespace Activations {
    // ReLU activation function
    static LambdaActivation ReLU(
        [](float x) { return std::max(0.0f, x); },           // Forward: ReLU
        [](float x) { return x > 0.0f ? 1.0f : 0.0f; }       // Backward: ReLU derivative
    );

    // Tanh activation function
    static LambdaActivation Tanh(
        [](float x) { return std::tanh(x); },                // Forward: Tanh
        [](float x) { float tanhx = std::tanh(x); return 1.0f - tanhx * tanhx; }  // Backward: Tanh derivative
    );

    // Sigmoid activation function
    static LambdaActivation Sigmoid(
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },  // Forward: Sigmoid
        [](float x) { float sigx = 1.0f / (1.0f + std::exp(-x)); return sigx * (1.0f - sigx); }  // Backward: Sigmoid derivative
    );
}
