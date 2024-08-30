#pragma once
#include <memory>
#include "ActivationFunction.hpp"

namespace Activations {

class ActivationWrapper {
private:
    // Pointer to hold either ActivationFunctionVectorLambda or ElementWiseActivation
    std::shared_ptr<BaseActivation> activation_;

public:
    // Default constructor - does nothing
    ActivationWrapper() : activation_(nullptr) {}

    // Constructor to initialize with ElementWiseActivation
    ActivationWrapper(const std::shared_ptr<ElementWiseActivation>& lambdaActivation)
        : activation_(lambdaActivation) {}

    // Constructor to initialize with ActivationFunctionVectorLambda
    ActivationWrapper(const std::shared_ptr<Tensor1DWiseActivation>& vectorLambdaActivation)
        : activation_(vectorLambdaActivation) {}

    // Function call operator to handle single input
    std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& input) {
        if (activation_) {
            // Use dynamic_cast to check if it is ElementWiseActivation
            auto lambdaActivation = std::dynamic_pointer_cast<ElementWiseActivation>(activation_);
            if (lambdaActivation) {
                return (*lambdaActivation)(input);
            }
        }
        return input;  // Do nothing if no valid activation is present
    }

    // Function call operator to handle multiple inputs
    Tensor1D operator()(const Tensor1D& inputs) {
        if (activation_) {
            // Use dynamic_cast to check if it is ElementWiseActivation
            auto lambdaActivation = std::dynamic_pointer_cast<ElementWiseActivation>(activation_);
            if (lambdaActivation) {
                return (*lambdaActivation)(inputs);
            }

            // Use dynamic_cast to check if it is Tensor1DWiseActivation
            auto vectorLambdaActivation = std::dynamic_pointer_cast<Tensor1DWiseActivation>(activation_);
            if (vectorLambdaActivation) {
                return (*vectorLambdaActivation)(inputs);
            }
        }
        return inputs;  // Do nothing if no valid activation is present
    }
};

} // namespace Activations
