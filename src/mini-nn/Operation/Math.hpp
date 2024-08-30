#pragma once
#include <memory>
#include "../Tensor/Tensor.hpp"

namespace Math
{
    /// @brief Sum the value of all tensor into a single Value
    /// @param tensor 
    /// @return ptr on the sum
    std::shared_ptr<Value> reduceSum(Tensor& tensor);
    
    /// @brief raise a tensor to a desired power (elementwise)
    /// @param tensor 
    /// @return a Tensor of
    Tensor pow(Tensor& tensor, int exponent);

} // namespace Math
