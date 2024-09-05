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

        // get bs
        int batchSize = input.dim()[0];

        if (!graphBuilded) {
            graphBuilded = true;
            input_ = Tensor::zeros(input[0].dim());
            std::vector<int> outshape;
            outshape.push_back(batchSize);
            for (auto val : layers_[layers_.size() - 1]->shape()) {
                outshape.push_back(val);
            }
            output_ = Tensor::zeros(outshape);

            Tensor x = input_;
            for (auto& layer : layers_) {
                x = layer->forward(x);
            }
            
            for (int i = 0 ; i < batchSize ; ++i){
                auto cloned = Math::cloneWithGraph(x);
                output_.assign(i, cloned);
            }

            // build graph from x, we can just call forward on output afterward
            computeGraph_ = Gradient::reverseTopologicalOrder(x);
        }

        for (int i = 0 ; i < batchSize ; ++i) {
            auto singleInput = input[i];
            input_.setValueLike(singleInput);

            // call compute graph (from input to x)
            for (int j = computeGraph_.size() - 1 ; j >= 0 ; --j) {
                computeGraph_[j]->forward();
            }

            // compute x to batch output
            for(auto& val : output_[i]) {
                val->forward();
            }
        }
        auto v = output_.getValues();
        return output_;
    }
};