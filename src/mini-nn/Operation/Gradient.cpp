#include "Gradient.hpp"

namespace Gradient
{
    std::vector<std::shared_ptr<Value>> topologicalOrder(const std::shared_ptr<Value>& root) {
        std::vector<std::shared_ptr<Value>> sorted;
        std::unordered_set<std::shared_ptr<Value>> visited;

        std::function<void(const std::shared_ptr<Value>&)> dfs = [&](const std::shared_ptr<Value>& node) {
            if (visited.count(node)) return;
            visited.insert(node);
            for (const auto& child : node->getChildren()) {
                dfs(child);
            }
            sorted.push_back(node);  // Add node to the sorted list after visiting all children
        };

        dfs(root);
        return std::move(sorted);
    }

    std::vector<std::shared_ptr<Value>> reverseTopologicalOrder(const std::shared_ptr<Value>& root) {
        std::vector<std::shared_ptr<Value>> nodes = topologicalOrder(root);
        std::reverse(nodes.begin(), nodes.end());
        return std::move(nodes);
    }

    std::vector<std::shared_ptr<Value>> reverseTopologicalOrder(Tensor& output) {
        if (output.size() != 1) {
            auto rootNode = Math::reduceSum(output);
            return reverseTopologicalOrder(rootNode);
        } else {
            return reverseTopologicalOrder(output.mat()[0]);
        }
    }


    void backward(std::vector<std::shared_ptr<Value>>& gradientNodes) {
        gradientNodes[0]->accumulateGrad(1.0f); // root node must be 0
        for (auto& node: gradientNodes) {
            node->backward();
        }
    }


    void zeroGrad(std::vector<std::shared_ptr<Value>>& gradientNodes) {
        for (auto& node : gradientNodes){
            node->zeroGrad();
        }
    }


    void derefGraph(std::vector<std::shared_ptr<Value>>& gradientNodes) {
        for (auto& node : gradientNodes){
            node->derefGraph();
        }
    }


    int clipGrad(std::vector<std::shared_ptr<Value>>& gradientNodes, float max) {
        if (max <= 0) {
            throw std::runtime_error("Max cannot be equal or less than 0");
        }

        int clipped = 0;
        for (auto& node : gradientNodes) {
            float gradient = node->getGrad();
            if (std::abs(gradient) > max) {
                clipped++;
                if (gradient > 0) {
                    node->setGradient(max);
                } else {
                    node->setGradient(-max);
                }
            }
        }

        return clipped;
    }

    void noiseGrad(std::vector<std::shared_ptr<Value>>& gradientNodes, float ratio) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0f, 0.01f);
        for (auto& node : gradientNodes) {
            float noise = dis(gen);
            node->accumulateGrad(noise / ratio);
        }
    }
} // namespace Gradient