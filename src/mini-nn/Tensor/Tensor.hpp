#pragma once
#include <vector>
#include <random>
#include <memory>
#include "../Values/Value.hpp"

class Tensor {
private:
    std::vector<std::shared_ptr<Value>> data_;
    int total_size_;
    std::vector<int> dimensions_;

    // private function to compute the 1D index from ND index
    int computeIndex(const std::vector<int>& indices) const;

public:
    Tensor() {};

    Tensor(const std::vector<int>& dims);

    Tensor(const std::vector<int>& dims, std::vector<std::shared_ptr<Value>> data);

    std::shared_ptr<Value>& operator()(const std::vector<int>& indices);

    const std::shared_ptr<Value>& operator()(const std::vector<int>& indices) const;

    Tensor operator[](int index);

    Tensor slice(int start, int end, int axis = 0);

    void fill(float value);

    const std::vector<std::shared_ptr<Value>>& mat() const;
    std::vector<std::shared_ptr<Value>>& mat();

    int rank() const;

    const std::vector<int>& dim() const;

    int size() const;

    void reshape(const std::vector<int>& new_dims);

    void flatten();

    void display() const;

    void setValueLike(Tensor& tensor);

    void assign(int index, Tensor& tensor);

    std::vector<float> getValues();


    // FOR ITERATOR

    auto begin() { return data_.data(); }
    auto end() { return data_.data() + data_.size(); }
    auto begin() const { return data_.data(); }
    auto end() const { return data_.data() + data_.size(); }


    // Public static factory methods
    static Tensor ones(const std::vector<int>& dims);
    static Tensor zeros(const std::vector<int>& dims);
    static Tensor random(const std::vector<int>& dims, float min = 0.0f, float max = 1.0f);
    static Tensor randn(const std::vector<int>& dims, float mean = 0.0f, float stddev = 1.0f);
};
