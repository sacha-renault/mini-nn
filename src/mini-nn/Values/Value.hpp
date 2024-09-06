#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#include "NodesTypes.hpp"


class Value : public std::enable_shared_from_this<Value> {
protected:
    float data_;
    float grad_;
    NodeTypes type_;
    std::unordered_set<std::shared_ptr<Value>> children_; // Children of the current Value in the computational graph
    std::vector<std::function<void()>> backwards_;                      // Backward function for backpropagation
    std::vector<std::function<void()>> forwards_;                       // forward function for optimized computation

public:

    // Constructor
    Value(float data)
        : data_(data), grad_(0.0f), children_(),
          backwards_(), forwards_(), type_(NodeTypes::ANY) { }

    static std::shared_ptr<Value> create(float data) {
        return std::make_shared<Value>(data);
    }

    // Getter for data and gradient
    float getData() const { return data_; }
    float getGrad() const { return grad_; }
    std::unordered_set<std::shared_ptr<Value>>& getChildren() { return children_; }
    void updateData(float stepSize) {
        data_ -= grad_ * stepSize;
    }

    // Setter for gradient
    void accumulateGrad(float grad) { grad_ += grad; }
    void setGradient(float grad) { grad_ = grad; }

    // Setter for value
    void setValue(float val) { data_ = val; }
    void setType(NodeTypes type) { type_ = type; }
    NodeTypes getType() { return type_; }

    // Set the backward function
    void addBackward(std::function<void()> func) { backwards_.push_back(func); }
    void addForward(std::function<void()> func) { forwards_.push_back(func); }

    // Add a child
    void addChild(const std::shared_ptr<Value>& child) {
        children_.insert(child);
    }

    // String representation for debugging
    std::string toString() const;

    // Apply a binary operator and create a new Value object
    std::shared_ptr<Value> applyOperator(const std::shared_ptr<Value>& other, std::function<float(float, float)> op_func);
    std::shared_ptr<Value> add(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> sub(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> times(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> div(const std::shared_ptr<Value>& other);

    // Backward pass initialization
    void backward();
    void forward();
    void zeroGrad();
    void derefGraph();
};
