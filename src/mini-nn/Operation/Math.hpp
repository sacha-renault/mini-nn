#pragma once
#include <memory>
#include "../Tensor/Tensor.hpp"

namespace Math
{
    /// @brief Sum the value of all tensor into a single Value
    /// @param tensor 
    /// @return ptr on the sum
    std::shared_ptr<Value> reduceSum(Tensor& tensor);

} // namespace Math
