#pragma once
#include <vector>
#include <memory>
#include "../Values/Value.hpp"

namespace Gradient
{
    std::vector<std::shared_ptr<Value>> getReverseTopologicalSort(const std::shared_ptr<Value>& root);
    
} // namespace Gradient

    