#include "Value.hpp"
#include "../Operation/Gradient.hpp"

std::string Value::toString() const {
    std::string rpr = "<Value=" + std::to_string(data_); // data value
    rpr += " ; Grad=" +  std::to_string(grad_); // grad value
    rpr += ">";
    return rpr;
}

std::shared_ptr<Value> Value::applyOperator(const std::shared_ptr<Value>& other, std::function<float(float, float)> op_func) {
    // Compute the new data value
    float newData = op_func(data_, other->getData());
    
    // Create a new Value object
    auto result = std::make_shared<Value>(newData);

    // Add the current object and `other` as children
    result->addChild(shared_from_this());
    result->addChild(other);

    return result;
}

void Value::backward() {
    if (backward_) {
        backward_();
    }
}

std::shared_ptr<Value> Value::add(const std::shared_ptr<Value>& other){
    std::shared_ptr<Value> out = applyOperator(other, [](float a, float b) { return a + b; });
    out->setBackward([out, other, this]() { 
        other->accumulateGrad(out->getGrad());
        this->accumulateGrad(out->getGrad());
    });
    return out;
}

std::shared_ptr<Value> Value::times(const std::shared_ptr<Value>& other){
    std::shared_ptr<Value> out = applyOperator(other, [](float a, float b) { return a * b; });
    out->setBackward([out, other, this]() { 
        other->accumulateGrad(out->getGrad() * this->data_);
        this->accumulateGrad(out->getGrad() * other->getData());
    });
    return out;
}

std::shared_ptr<Value> Value::sub(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = applyOperator(other, [](float a, float b) { return a - b; });
    out->setBackward([out, other, this]() { 
        other->accumulateGrad(-out->getGrad()); // Negative gradient for subtraction
        this->accumulateGrad(out->getGrad());
    });
    return out;
}

std::shared_ptr<Value> Value::div(const std::shared_ptr<Value>& other) {
    std::shared_ptr<Value> out = applyOperator(other, [](float a, float b) { return a / b; });
    out->setBackward([out, other, this]() { 
        other->accumulateGrad(-out->getGrad() * this->data_ / (other->getData() * other->getData()));
        this->accumulateGrad(out->getGrad() / other->getData());
    });
    return out;
}



void Value::derefGraph() {
    backward_ = nullptr; // deref the lambda function and free memory
    zeroGrad();
    children_.clear();  // clean all refs to child, since it's share_ptr, 
                        // any shared_ptr that doesn't have owner will be free
}

void Value::zeroGrad() {
    grad_ = 0.0f;
}