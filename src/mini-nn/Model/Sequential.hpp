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

        if (!graphBuilded) {
            int batchSize = input.dim()[0];
            graphBuilded = true;
            input_ = input; // TODO, real copy of tensor input

            std::vector<int> outshape;
            outshape.push_back(batchSize);
            for (auto val : layers_[layers_.size() - 1]->shape()) {
                outshape.push_back(val);
            }
            output_ = Tensor::zeros(outshape);

            for (int i = 0 ; i < batchSize ; ++i){
                Tensor x = input_[i];
                for (auto& layer : layers_) {
                    x = layer->forward(x);
                }
                output_.assign(i, x);
            }

            computeGraph_ = Gradient::reverseTopologicalOrder(output_);
        }

        else {
            input_.setValueLike(input);
            for (int i = computeGraph_.size() - 1 ; i >= 0 ; --i) {
                computeGraph_[i]->forward();
            }
        }
        return output_;
    }
};