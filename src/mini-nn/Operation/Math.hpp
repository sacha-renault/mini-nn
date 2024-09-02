#pragma once
#include <memory>
#include "../Tensor/Tensor.hpp"

namespace Math
{
    /// @brief Sum the value of all tensor into a single Value
    /// @param tensor 
    /// @return ptr on the sum
    std::shared_ptr<Value> reduceSum(Tensor& tensor);

    /// @brief calc the mean value of all tensor into a single Value
    /// @param tensor 
    /// @return ptr on the sum
    std::shared_ptr<Value> reduceMean(Tensor& tensor);
    

    /// @brief raise a Value to a desired power
    /// @param tensor 
    /// @return tensor
    std::shared_ptr<Value> pow(std::shared_ptr<Value> base, int exponent);


    /// @brief raise a tensor to a desired power (elementwise)
    /// @param tensor 
    /// @return tensor
    Tensor pow(Tensor& tensor, int exponent);

} // namespace Math
