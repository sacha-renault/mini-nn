#pragma once
#include <memory>
#include "Model.hpp"
#include "../Layers/Layer.hpp"

class Sequential : public Model {
protected:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Value>> computeGraph;
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
        if (!graphBuilded) {
            graphBuilded = true;
            input_ = input; // TODO, real copy of tensor input
            auto x = input_;
            for (auto& layer : layers_) {
                x = layer->forward(x);
            }
            output_ = x;
            computeGraph = Gradient::reverseTopologicalOrder(output_);
        } else {
            input_.setValueLike(input);
            for (int i = computeGraph.size() - 1 ; i >= 0 ; --i) {
                computeGraph[i]->forward();
            }
        }
        return output_;
    }
};