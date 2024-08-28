#pragma once
#include <string>
#include <vector>
#include <functional>

enum class Op {
    add, sub, mul, div 
};

class Value {
protected:
    float data_;
    float grad_;
    std::vector<Value> childrens_;
    Op op_;

public:
    // Constructor
    Value(float data) 
        : data_(data), grad_(0.0f) {}
    Value(float data, std::vector<Value>& childrens, Op op) 
        : data_(data), childrens_(childrens), op_(op), grad_(0.0f) {}

    // Getter / setter
    float getData() const { return data_; }
    void setGrad(float grad) { grad_ = grad; }

    // public method
    std::string toString() const;
    Value applyOperator(const Value& other, std::function<float(float, float)> op_func, Op op);
    void backpropagation();

    // operator overload
    Value operator+(const Value& other);
    Value operator-(const Value& other);
    Value operator*(const Value& other);
    Value operator/(const Value& other);
};