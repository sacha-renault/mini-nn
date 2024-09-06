#include "Dense.hpp"

namespace Layers
{
    const Tensor& Dense::forward(Tensor& inputs) {
        // Assuming inputs is of shape [batch_size, input_dim]
        int batchSize = inputs.dim()[0];
        int inputDim = inputs.dim()[1];

        // Initialize outputs to hold the result; output shape is [batchSize, neurons_.size()]
        outputs_ = Tensor::zeros({batchSize, (int)neurons_.size()});

        // Iterate over each neuron
        int i = 0;
        for (auto& neuron : neurons_) {
            // Perform a dot product of inputs with the neuron's weights and add the bias
            // Assuming neurons_[i].forward(inputs) is implemented to handle batch input
            // If not, we might need to modify that function as well
            Tensor neuronOutput = neuron.forward(inputs);  // Output shape is [batchSize, 1]

            // Assign the output of this neuron to the i-th column of the outputs_ tensor
            for(int b_num = 0 ; b_num < neuronOutput.size() ; ++b_num) {
                outputs_({b_num, i}) = neuronOutput({b_num});
            }

            // increment
            i++;
        }

        // Apply the activation function to the entire output, if one is defined
        if (func_) {
            outputs_ = func_(outputs_);
        }

        return outputs_;  // Return the batch outputs
    }

    Tensor Dense::getParameters() {
        // First calculate the number of weight
        int total_weights = 0;
        for (auto& neuron : neurons_) {
            total_weights += neuron.getWeights().size();  // Count total number of weights
            total_weights += 1; // also has a bias
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

