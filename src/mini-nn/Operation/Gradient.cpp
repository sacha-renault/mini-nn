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
            return reverseTopologicalOrder(output({0}));
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
} // namespace Gradient