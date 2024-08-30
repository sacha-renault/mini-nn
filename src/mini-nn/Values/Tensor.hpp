#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "../Values/Value.hpp"

template<int Rank>
class Tensor {
private:
    Eigen::Tensor<std::shared_ptr<Value>, Rank> data_;

public:
    // Default constructor
    Tensor() {}

    // Constructor with dimensions
    Tensor(const std::array<Eigen::Index, Rank>& dimensions) : data_(dimensions) {}

    // Access element by index
    std::shared_ptr<Value>& operator()(const std::array<Eigen::Index, Rank>& indices) {
        return data_(indices);
    }

    const std::shared_ptr<Value>& operator()(const std::array<Eigen::Index, Rank>& indices) const {
        return data_(indices);
    }

    // Resize tensor
    void resize(const std::array<Eigen::Index, Rank>& new_dimensions) {
        data_.resize(new_dimensions);
    }

    // Fill tensor with a specific value
    void fill(const std::shared_ptr<Value>& value) {
        data_.setConstant(value);
    }

    // Get dimensions of the tensor
    std::array<Eigen::Index, Rank> dimensions() const {
        std::array<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) {
            dims[i] = data_.dimension(i);
        }
        return dims;
    }

    // Apply a function to all elements
    template <typename Func>
    void apply(Func func) {
        data_ = data_.unaryExpr(func);
    }

    // Sum all elements (assuming std::shared_ptr<Value> has a sum operation)
    std::shared_ptr<Value> sum() const {
        auto result = std::make_shared<Value>(0.0f); // Assume Value has a constructor that takes a float
        for (int i = 0; i < data_.size(); ++i) {
            result = result->add(data_.data()[i]);
        }
        return result;
    }

    // Get the underlying Eigen::Tensor
    Eigen::Tensor<std::shared_ptr<Value>, Rank>& data() {
        return data_;
    }

    const Eigen::Tensor<std::shared_ptr<Value>, Rank>& data() const {
        return data_;
    }

    // Get the rank
    int rank() const { return Rank; }

    // Get dimensions of the tensor
    std::array<Eigen::Index, Rank> dim() const {
        std::array<Eigen::Index, Rank> dims;
        for (int i = 0; i < Rank; ++i) {
            dims[i] = data_.dimension(i);
        }
        return dims;
    }

    Eigen::Index size() const {
        return data_.size();
    }

    // Display tensor for debugging
    void display() const {
        std::cout << "Tensor: ";
        for (int i = 0; i < data_.size(); ++i) {
            std::cout << data_.data()[i]->getData() << " ";  // Assuming Value has a get() method
        }
        std::cout << std::endl;
    }

    // FOR ITERATOR
    // Iterator support for 1D tensors
    auto begin() {
        static_assert(Rank == 1, "Iteration is only supported for 1D tensors.");
        return data_.data();  // Returns pointer to the start of the data
    }

    auto end() {
        static_assert(Rank == 1, "Iteration is only supported for 1D tensors.");
        return data_.data() + data_.size();  // Returns pointer to the end of the data
    }

    auto begin() const {
        static_assert(Rank == 1, "Iteration is only supported for 1D tensors.");
        return data_.data();  // Returns pointer to the start of the data
    }

    auto end() const {
        static_assert(Rank == 1, "Iteration is only supported for 1D tensors.");
        return data_.data() + data_.size();  // Returns pointer to the end of the data
    }
};


using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;
using Tensor3D = Tensor<3>;
using Tensor4D = Tensor<4>;
using Tensor5D = Tensor<5>;
using Tensor6D = Tensor<6>;
using Tensor7D = Tensor<7>;
