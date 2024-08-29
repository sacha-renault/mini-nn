#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>

struct GradVal {
    float lgrad;
    float rgrad;
    GradVal(float l, float r) : lgrad(l), rgrad(r) { }
};

class Value {
protected:
    float data_;
    float grad_;
    std::vector<Value*> childrens_;
    std::function<void()> backward_;
    void _backpropagation();

public:
    // Constructors
    Value(float data) 
        : data_(data), grad_(0.0f), childrens_() {}
    // Value(float data, const std::vector<Value*>& children) 
    //     : data_(data), grad_(0.0f), childrens_(children) {}
    Value(float data, const std::vector<Value*> children) 
        : data_(data), grad_(0.0f), childrens_(std::move(children)) {}

    // Getter / setter
    float getData() const { return data_; }
    void setGrad(float grad) { grad_ += grad; }
    float getGrad() const { return grad_; }
    void setBackward(std::function<void()> func) { backward_ = func; } 
    void clearBackward() { backward_ = nullptr; }

    // public method
    std::string toString() const;
    Value applyOperator(Value& other, std::function<float(float, float)> op_func);
    void backpropagation();

    // operator overload
    Value operator+(Value& other);
    Value operator*(Value& other);
};

Value sumManyValue(std::vector<Value*>& others);