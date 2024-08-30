#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "../Values/Value.hpp"

class Tensor {
private:
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> data_;
    std::vector<Eigen::Index> dimensions_;
    Eigen::Index total_size_;

    // private function to compute the 1D index from ND index
    Eigen::Index computeIndex(const std::vector<Eigen::Index>& indices) const {
        Eigen::Index index = 0;
        Eigen::Index multiplier = 1;
        for (int i = dimensions_.size() - 1; i >= 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= dimensions_[i];
        }
        return index;
    }

public:
    Tensor() {};

    // Constructor that accepts dimensions
    Tensor(const std::vector<Eigen::Index>& dims)
        : dimensions_(dims), total_size_(1) {
        for (const auto& dim : dimensions_) {
            total_size_ *= dim;
        }
        data_.resize(total_size_);
    }

    // Access element by multi-dimensional indices
    std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices) {
        return data_(computeIndex(indices));
    }
    const std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices) const {
        return data_(computeIndex(indices));
    }

    // Fill tensor with a specific value
    void fill(const std::shared_ptr<Value>& value) {
        data_.setConstant(value);
    }

    // Get the underlying Eigen::Matrix
    const Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>& data() const {
        return data_;
    }

    // Get the rank
    int rank() const { return dimensions_.size(); }

    // Get dimensions of the tensor
    std::vector<Eigen::Index> dim() const {
        return dimensions_;
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
    auto begin() { return data_.data(); }
    auto end() { return data_.data() + data_.size(); }
    auto begin() const { return data_.data(); }
    auto end() const { return data_.data() + data_.size(); }
};
