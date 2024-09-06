#pragma once
#include <memory>
#include "Model.hpp"
#include "../Layers/Layer.hpp"
#include "../Operation/Gradient.hpp"

class Sequential : public Model {
protected:
    std::vector<std::shared_ptr<Layer>> layers_;
    std::vector<std::shared_ptr<Value>> computeGraph_;
    Tensor input_;
    Tensor output_;
    bool graphBuilded;
public:
    Sequential();

    /// @brief Push back a layer into the layers list by ref
    /// @param layer 
    void addLayer(const std::shared_ptr<Layer> layer);

    /// @brief Get all parameters of the model
    /// @return params in the model
    std::vector<std::shared_ptr<Value>> getParameters() override;

    /// @deprecated
    void update(float lr) override { }; // we actually don't use this

    /// @brief 
    /// @param input tensor of shape (batch_size, *input_size)
    /// @return output tensor of shape (batch_size, *output_size)
    const Tensor& forward(Tensor& input) override;
};