#pragma once
#include <vector>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#include "NodesTypes.hpp"
#include "NodeOperation.hpp"


class Value : public std::enable_shared_from_this<Value> {
protected:
    float data_;
    float grad_;
    NodeTypes type_;
    std::vector<std::shared_ptr<Value>> children_; // Children of the current Value in the computational graph
    std::unique_ptr<NodeOperation::Operation> operation_;

public:

    // Constructor
    Value(float data)
        : data_(data), grad_(0.0f), children_(),
          type_(NodeTypes::ANY), operation_(nullptr) { }

    Value(float data, std::unique_ptr<NodeOperation::Operation> op)
        : data_(data), grad_(0.0f), children_(),
          type_(NodeTypes::ANY), operation_(std::move(op)) { }

    static std::shared_ptr<Value> create(float data) {
        return std::make_shared<Value>(data);
    }

    static std::shared_ptr<Value> create(float data, std::unique_ptr<NodeOperation::Operation> op) {
        return std::make_shared<Value>(data, std::move(op));
    }

    // Getter for data and gradient
    float getData() const { return data_; }
    float getGrad() const { return grad_; }
    std::vector<std::shared_ptr<Value>>& getChildren() { return children_; }


    // Setter for gradient
    void accumulateGrad(float grad) { grad_ += grad; }
    void setGradient(float grad) { grad_ = grad; }

    // Setter for value
    void setValue(float val) { data_ = val; }
    void setType(NodeTypes type) { type_ = type; }
    NodeTypes getType() { return type_; }

    // Add a child
    void addChild(const std::shared_ptr<Value>& child) {
        children_.push_back(child);
    }

    // String representation for debugging
    std::string toString() const;

    // Backward pass initialization
    void backward();
    void forward();
    void zeroGrad();
    void derefGraph();
};
