#pragma once
#include <vector>
#include <memory>
#include "../Values/Value.hpp"

namespace Gradient
{
    /// @brief Get all the node sorted in topoligical order
    /// @param root output node
    /// @return unaccumulated gradient
    std::vector<std::shared_ptr<Value>> getReverseTopologicalSort(const std::shared_ptr<Value>& root);
    
} // namespace Gradient

    