#pragma once
#include <memory>
#include "Model.hpp"
#include "../Layers/Layer.hpp"

class Sequential : public Model {
protected:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Value>> computeGraph_;
    Tensor input_;
    Tensor output_;
    bool graphBuilded;
public:
    Sequential() : layers_(), graphBuilded(false) {}

    // Add layers
    void addLayer(const std::shared_ptr<Layer>& layer) {
        layers_.push_back(layer);
    }
    void addLayer(std::shared_ptr<Layer>&& layer) {
        layers_.push_back(std::move(layer));
    }

    std::vector<std::shared_ptr<Value>> getParameters() override {
        std::vector<std::shared_ptr<Value>> params;
        for(auto& layer : layers_) {
            for(auto& param : layer->getParameters()){
                params.push_back(param);
            }
        }
        return params;
    };


    void update(float lr) override {
        for(auto& param: getParameters()){
            param->updateData(lr);
        }
    }

    const Tensor& forward(Tensor& input) override {
        if (input.rank() < 2) {
            throw std::runtime_error("input of rank 1 cannot be batched input");
        }

        // Batch processing: input is [batch_size, ...]
        int batchSize = input.dim()[0];

        // Only build the graph once
        if (!graphBuilded) {
            graphBuilded = true;

            input_ = Tensor::zeros(input.dim());

            // Define the computation for a single input, but process the whole batch
            Tensor x = input_;
            for (auto& layer : layers_) {
                x = layer->forward(x);  // Each layer handles batched inputs
            }

            output_ = x;  // Final output is batched
            computeGraph_ = Gradient::reverseTopologicalOrder(x);  // Single graph
        }

        // assign new values to input
        input_.setValueLike(input);

        // No need to loop over each batch element; process the entire batch at once
        for (int j = computeGraph_.size() - 1 ; j >= 0 ; --j) {
            computeGraph_[j]->forward();  // Forward pass for the entire batch
        }

        return output_;
    }
};