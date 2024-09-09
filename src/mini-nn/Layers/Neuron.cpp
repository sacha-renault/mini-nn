#include "Neuron.hpp"

Neuron::Neuron(int num_inputs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Initialize weights
    wi_ = Tensor::randn({ num_inputs });

    // Create the bias point
    bias_ = Value::create(dist(gen));  // Initialize bias

    // Init types
    bias_->setType(NodeTypes::BIAS);
    for(auto& val : wi_) {
        val->setType(NodeTypes::WEIGHT);
    }
}

Tensor& Neuron::forward(Tensor& xi) {
    // Assert that Tensor has rank 2 (batch size, input dimension)
    if (xi.rank() != 2) {
        throw std::invalid_argument("Input tensor must be rank 2 for batched input (batch_size, input_dim).");
    }

    // Extract batch size and input dimension
    int batchSize = xi.dim()[0];
    int inputDim = xi.dim()[1];

    // Assert dimensions are compatible
    if (inputDim != wi_.dim()[0]) {
        throw std::invalid_argument("Input size must match the number of weights.");
    }

    // Initialize the output tensor to hold the neuron outputs for each example in the batch
    Tensor outputs({ batchSize });

    // Compute the weighted inputs for each data in batch
    for (int i = 0; i < batchSize; ++i) {
        // Compute the weighted sum for the i-th example
        Tensor xi_batch = xi[i];
        Tensor xiwi_batch({ inputDim });

        for (int j = 0; j < inputDim; ++j) {
            xiwi_batch({j}) = xi_batch({j}) * wi_({j});
        }

        // Sum all weighted inputs and add bias for this batch element
        auto xnwn_batch = Math::reduceSum(xiwi_batch);
        auto output_batch = xnwn_batch + bias_;  // Add bias

        // Store the output for this batch element
        outputs({i}) = output_batch;
    }

    // Store the batch outputs
    output_ = outputs;

    return output_;
}