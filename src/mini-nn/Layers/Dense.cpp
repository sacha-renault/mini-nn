#include "Dense.hpp"

namespace Layers
{
    Tensor<1>& Dense::forward(Tensor<1>& inputs) {
        for (int i = 0; i < neurons_.size(); ++i) {
            outputs_({i}) = neurons_[i].forward(inputs);  // Store each neuron's output
        }
        
        outputs_ = func_(outputs_);

        return outputs_;
    }

    void Dense::backward() {
        // Ensure output is a single element
        if (outputs_.dim()[0] == 1) {
            outputs_({0})->backward();
        } 
        else { // Case there isn't a single element, we sum first to start backprop on a single node
            auto node = outputs_.sum();
            node->backward();
        }
    }

    Tensor<1> Dense::getParameters() {
        // First calculate the number of weight
        int total_weights = 0;
        for (auto& neuron : neurons_) {
            total_weights += neuron.getWeights().size();  // Count total number of weights
        }

        // Create a Tensor to hold all the parameters
        Tensor<1> params({total_weights});

        // Copy each neuron's weights into the params Tensor
        int index = 0;
        for (auto& neuron : neurons_) {
            Tensor<1> weights = neuron.getWeights();
            for (int i = 0; i < weights.dimensions()[0]; ++i) {
                params({index++}) = weights({i});
            }
        }
        return params;
    }

    Tensor<1> Dense::getBiases() {
        int n = neurons_.size();
        Tensor<1> params({n}); // as many biases as neurone
        for (int i = 0 ; i < n ; ++i) {
            auto weights = neurons_[i].getWeights();  // Get weights of each neuron
            params({i}) = neurons_[i].getBias();
        }
        return params;
    }
}

