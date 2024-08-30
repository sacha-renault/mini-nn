#pragma once
#include <vector>
#include <memory>
#include "Math.hpp"
#include "../Values/Value.hpp"
#include "../Tensor/Tensor.hpp"

namespace Gradient
{
    /// @brief Get all the node sorted in topoligical order
    /// @param root output node
    /// @return unaccumulated gradient
    std::vector<std::shared_ptr<Value>> getGraphNodes(const std::shared_ptr<Value>& root);

    /// @brief Get all the node sorted in topoligical order
    /// @param root output tensor
    /// @return unaccumulated gradient
    std::vector<std::shared_ptr<Value>> getGraphNodes(Tensor& output);

    /// @brief Propagate the gradient through all the graph
    /// @param all the nodes of the computional graph
    void backward(std::vector<std::shared_ptr<Value>> gradientNodes);


    /// @brief reset the gradient of all nodes to 0.0;
    /// it doesn't destroy the compute graph
    /// @param gradientNodes 
    void zeroGrad(std::vector<std::shared_ptr<Value>>& gradientNodes);


    /// @brief reset the gradient of all nodes to 0.0, free children and delete backward lambdas;
    /// it does destroy the compute graph
    /// @param gradientNodes 
    void derefGraph(std::vector<std::shared_ptr<Value>>& gradientNodes);
    
} // namespace Gradient

    