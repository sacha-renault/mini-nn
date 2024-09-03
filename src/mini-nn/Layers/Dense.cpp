#include "Dense.hpp"

namespace Layers
{
    const Tensor& Dense::forward(Tensor& inputs) {
        for (int i = 0; i < neurons_.size(); ++i) {
            outputs_({i}) = neurons_[i].forward(inputs);  // Store each neuron's output
        }

        activationOutputs_ = func_(outputs_);

        return activationOutputs_;
    }

    // void Dense::backward() {
    //     // Ensure output is a single element
    //     if (outputs_.dim()[0] == 1) {
    //         outputs_({0});
    //     }
    //     else { // Case there isn't a single element, we sum first to start backprop on a single node
    //         auto node = Math::reduceSum(outputs_);
    //         node->backward();
    //     }
    // }

    Tensor Dense::getParameters() {
        // First calculate the number of weight
        int total_weights = 0;
        for (auto& neuron : neurons_) {
            total_weights += neuron.getWeights().size();  // Count total number of weights
            total_weights += 1; // also as a bias
        }

        // Create a Tensor to hold all the parameters
        Tensor params({total_weights});

        // Copy each neuron's weights into the params Tensor
        int index = 0;
        for (auto& neuron : neurons_) {
            Tensor weights = neuron.getWeights();
            for (int i = 0; i < weights.dim()[0]; ++i) {
                params({index++}) = weights({i});
            }
            params({index++}) = neuron.getBias();
        }
        return params;
    }

    Tensor Dense::getBiases() {
        int n = neurons_.size();
        Tensor params({n}); // as many biases as neurone
        for (int i = 0 ; i < n ; ++i) {
            auto weights = neurons_[i].getWeights();  // Get weights of each neuron
            params({i}) = neurons_[i].getBias();
        }
        return params;
    }
}

