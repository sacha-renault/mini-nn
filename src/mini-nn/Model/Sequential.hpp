#pragma once
#include <memory>
#include "Model.hpp"
#include "../Layers/Layer.hpp"

class Sequential : public Model {
protected:
    std::vector<std::shared_ptr<Layer>> layers_;
public:
    Sequential() : layers_() {}

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
    Tensor forward(Tensor& input) override {
        auto x = input;
        for (auto& layer : layers_) {
            x = layer->forward(x);
        }
        return x;
    }
};