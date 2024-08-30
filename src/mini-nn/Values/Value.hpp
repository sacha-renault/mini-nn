#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <string>
#include <iostream>


class Value : public std::enable_shared_from_this<Value> {
protected:
    float data_;
    float grad_;
    std::vector<std::shared_ptr<Value>> children_; // Children of the current Value in the computational graph
    std::function<void()> backward_;               // Backward function for backpropagation

public:
    // Constructor
    Value(float data)
        : data_(data), grad_(0.0f), children_() { }

    static std::shared_ptr<Value> create(float data) {
        return std::make_shared<Value>(data);
    }

    // Getter for data and gradient
    float getData() const { return data_; }
    float getGrad() const { return grad_; }
    std::vector<std::shared_ptr<Value>>& getChildren() { return children_; }

    // Setter for gradient
    void accumulateGrad(float grad) { grad_ += grad; }

    // Set the backward function
    void setBackward(std::function<void()> func) { backward_ = func; }

    // Add a child
    void addChild(const std::shared_ptr<Value>& child) {
        children_.push_back(child);
    }

    // String representation for debugging
    std::string toString() const;

    // Apply a binary operator and create a new Value object
    std::shared_ptr<Value> applyOperator(const std::shared_ptr<Value>& other, std::function<float(float, float)> op_func);
    std::shared_ptr<Value> add(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> times(const std::shared_ptr<Value>& other);

    // Backward pass initialization
    void backward();
    void zeroGrad();

protected:
    // Internal method for recursive backpropagation
    void _backpropagation();
};
