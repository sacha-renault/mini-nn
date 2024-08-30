#include "Gradient.hpp"

namespace Gradient
{
    std::vector<std::shared_ptr<Value>> getReverseTopologicalSort(const std::shared_ptr<Value>& root) {
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
        std::reverse(sorted.begin(), sorted.end());
        return std::move(sorted);
    }
} // namespace Gradient