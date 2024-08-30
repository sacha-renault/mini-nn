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
    Eigen::Index computeIndex(const std::vector<Eigen::Index>& indices) const;

public:
    Tensor() {};

    Tensor(const std::vector<Eigen::Index>& dims);

    std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices);

    const std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices) const;

    void fill(const std::shared_ptr<Value>& value);

    const Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>& data() const;

    int rank() const;

    std::vector<Eigen::Index> dim() const;

    Eigen::Index size() const;

    void reshape(const std::vector<Eigen::Index>& new_dims);

    void flatten();

    void display() const;

    // FOR ITERATOR

    auto begin() { return data_.data(); }
    auto end() { return data_.data() + data_.size(); }
    auto begin() const { return data_.data(); }
    auto end() const { return data_.data() + data_.size(); }
};
