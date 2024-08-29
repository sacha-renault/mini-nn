#include "Dense.hpp"

namespace Layers
{
    std::vector<std::shared_ptr<Value>>& Dense::forward(std::vector<std::shared_ptr<Value>>& inputs) {
        for (size_t i = 0; i < neurons_.size(); ++i) {
            outputs_[i] = neurons_[i].forward(inputs);  // Store each neuron's output
        }
        
        outputs_ = func_(outputs_);

        return outputs_;
    }

    void Dense::backward() {
        for (auto& out : outputs_) {
            out->backward();  // Perform backpropagation for each neuron's output
        }
    }

    std::vector<std::shared_ptr<Value>> Dense::getParameters() {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& neuron : neurons_) {
            auto weights = neuron.getWeights();  // Get weights of each neuron
            params.insert(params.end(), weights.begin(), weights.end());
        }
        return params;
    }

    std::vector<std::shared_ptr<Value>> Dense::getBiases() {
        std::vector<std::shared_ptr<Value>> params;
        for (auto& neuron : neurons_) {
            auto weights = neuron.getWeights();  // Get weights of each neuron
            params.push_back(neuron.getBias());
        }
        return params;
    }
}

