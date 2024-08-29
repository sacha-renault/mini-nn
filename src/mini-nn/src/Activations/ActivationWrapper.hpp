#pragma once
#include <memory>
#include "ActivationFunction.hpp"

namespace Activations {

class ActivationWrapper {
private:
    // Pointer to hold either ActivationFunctionVectorLambda or LambdaActivation
    std::shared_ptr<BaseActivation> activation_;

public:
    // Default constructor - does nothing
    ActivationWrapper() : activation_(nullptr) {}

    // Constructor to initialize with LambdaActivation
    ActivationWrapper(const std::shared_ptr<LambdaActivation>& lambdaActivation)
        : activation_(lambdaActivation) {}

    // Constructor to initialize with ActivationFunctionVectorLambda
    ActivationWrapper(const std::shared_ptr<LambdaActivationVector>& vectorLambdaActivation)
        : activation_(vectorLambdaActivation) {}

    // Function call operator to handle single input
    std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& input) {
        if (activation_) {
            // Use dynamic_cast to check if it is LambdaActivation
            auto lambdaActivation = std::dynamic_pointer_cast<LambdaActivation>(activation_);
            if (lambdaActivation) {
                return (*lambdaActivation)(input);
            }
        }
        return input;  // Do nothing if no valid activation is present
    }

    // Function call operator to handle multiple inputs
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        if (activation_) {
            // Use dynamic_cast to check if it is LambdaActivation
            auto lambdaActivation = std::dynamic_pointer_cast<LambdaActivation>(activation_);
            if (lambdaActivation) {
                return (*lambdaActivation)(inputs);
            }

            // Use dynamic_cast to check if it is LambdaActivationVector
            auto vectorLambdaActivation = std::dynamic_pointer_cast<LambdaActivationVector>(activation_);
            if (vectorLambdaActivation) {
                return (*vectorLambdaActivation)(inputs);
            }
        }
        return inputs;  // Do nothing if no valid activation is present
    }
};

} // namespace Activations
