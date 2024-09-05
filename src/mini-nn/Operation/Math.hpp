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

    /// @brief raise a tensor to a desired power (elementwise)
    /// @param tensor
    /// @return tensor
    Tensor pow(Tensor& t1, Tensor& t2);

    /// @brief element-wise sum
    /// @param t1
    /// @param t2
    /// @return sum of tensor elementwise
    Tensor ewSum(Tensor& t1, Tensor& t2);

    /// @brief element-wise subtraction
    /// @param t1
    /// @param t2
    /// @return subtraction of tensor elementwise
    Tensor ewSub(Tensor& t1, Tensor& t2);

    /// @brief element-wise multiplication
    /// @param t1
    /// @param t2
    /// @return multiplication of tensor elementwise
    Tensor ewMul(Tensor& t1, Tensor& t2);

    /// @brief element-wise division
    /// @param t1
    /// @param t2
    /// @return division of tensor elementwise
    Tensor ewDiv(Tensor& t1, Tensor& t2);

    /// @brief element-wise division
    /// @param t1
    /// @param t2
    /// @return division of tensor elementwise
    Tensor abs(Tensor& t1);

    /// @brief continue graph with same grad and value
    /// @param t1
    /// @return same tensor on grpah (different ref)
    Tensor cloneWithGraph(Tensor& t1);

} // namespace Math
