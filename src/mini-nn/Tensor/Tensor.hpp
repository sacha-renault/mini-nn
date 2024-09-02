#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include "../Values/Value.hpp"

class Tensor {
private:
    std::shared_ptr<Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>> data_;
    std::vector<Eigen::Index> dimensions_;
    Eigen::Index total_size_;
    
    // For subtensor
    Eigen::Index offset_;
    std::vector<Eigen::Index> strides_;

    // private function to compute the 1D index from ND index
    Eigen::Index computeIndex(const std::vector<Eigen::Index>& indices) const;

public:
    Tensor() {};

    Tensor(const std::vector<Eigen::Index>& dims);

    std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices);

    const std::shared_ptr<Value>& operator()(const std::vector<Eigen::Index>& indices) const;

    void fill(const std::shared_ptr<Value>& value);

    const Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>& mat() const;
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1>& mat();

    int rank() const;

    std::vector<Eigen::Index> dim() const;

    Eigen::Index size() const;

    void reshape(const std::vector<Eigen::Index>& new_dims);

    void flatten();

    void display() const;



    // FOR ITERATOR

    auto begin() { return data_->data(); }
    auto end() { return data_->data() + data_->size(); }
    auto begin() const { return data_->data(); }
    auto end() const { return data_->data() + data_->size(); }


    // Public static factory methods
    static Tensor ones(const std::vector<Eigen::Index>& dims);
    static Tensor zeros(const std::vector<Eigen::Index>& dims);
    static Tensor random(const std::vector<Eigen::Index>& dims, float min = 0.0f, float max = 1.0f);
    static Tensor randn(const std::vector<Eigen::Index>& dims, float mean = 0.0f, float stddev = 1.0f);
};
