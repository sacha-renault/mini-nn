#pragma once
#include <vector>
#include <memory>
#include <math.h>
#include <iostream>
#include "Math.hpp"
#include "../Values/Value.hpp"
#include "../Tensor/Tensor.hpp"

namespace Gradient
{
    /// @brief Get all the node sorted in topoligical order
    /// @param root output node
    /// @return all ordered nodes
    std::vector<std::shared_ptr<Value>> topologicalOrder(const std::shared_ptr<Value>& root);


    /// @brief Get all the node sorted in reverse topoligical order
    /// @param root output node
    /// @return all ordered nodes
    std::vector<std::shared_ptr<Value>> reverseTopologicalOrder(const std::shared_ptr<Value>& root);

    /// @brief Get all the node sorted in reverse topoligical order from a Tensor
    /// @param root output tensor
    /// @return all ordered nodes
    std::vector<std::shared_ptr<Value>> reverseTopologicalOrder(Tensor& output);

    /// @brief Propagate the gradient through all the graph
    /// @param all the nodes of the computional graph
    void backward(std::vector<std::shared_ptr<Value>>& gradientNodes);


    /// @brief reset the gradient of all nodes to 0.0;
    /// it doesn't destroy the compute graph
    /// @param gradientNodes
    void zeroGrad(std::vector<std::shared_ptr<Value>>& gradientNodes);


    /// @brief reset the gradient of all nodes to 0.0, free children and delete backward lambdas;
    /// it does destroy the compute graph
    /// @param gradientNodes
    void derefGraph(std::vector<std::shared_ptr<Value>>& gradientNodes);

    /// @brief clip gradient to a abs max value
    /// @param gradientNodes
    /// @return number of node where gradient was clipped
    int clipGrad(std::vector<std::shared_ptr<Value>>& gradientNodes, float max = 1.0f);

    /// @brief add noise ~N(0, 0.01) to gradient (to avoid local minima) / ratio
    /// @param gradientNodes
    /// @param ratio to divide the result of normal noise
    /// @return number of node where gradient was clipped
    void noiseGrad(std::vector<std::shared_ptr<Value>>& gradientNodes, float ratio);

} // namespace Gradient

