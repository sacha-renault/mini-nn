#include "Value.hpp"

std::string Value::toString() const {
    std::string rpr = "<Value=" + std::to_string(data_); // data value
    rpr += " ; Grad=" +  std::to_string(grad_); // grad value
    rpr += ">";
    return rpr;
}

Value Value::applyOperator(Value& other, std::function<float(float, float)> op_func) {
    // Create a vector to hold the children
    std::vector<Value*> children;
    children.push_back(this);    // Add `this` as a child
    children.push_back(&other);  // Add `other` as another child

    // create a new value obj with new childrens
    return Value(
        op_func(data_, other.getData()),  
        children);
}

Value Value::operator+(Value& other) {
    Value out = applyOperator(other, [](float a, float b) { return a + b; });
    out.setBackward([&out, &other, this]() { 
        other.setGrad(out.getGrad());
        this->setGrad(out.getGrad());
    });
    return out;
}

Value Value::operator*(Value& other) {
    Value out = applyOperator(other, [](float a, float b) { return a * b; });
    out.setBackward([&out, &other, this]() { 
        other.setGrad(out.getGrad() * this->data_);
        this->setGrad(out.getGrad() * other.getData());
    });
    return out;
}

void Value::backpropagation() {
    grad_ = 1.0f; // we set the grad to 1 on the last value.
    _backpropagation();
}

void Value::_backpropagation() {
    if (backward_) {
        backward_();
    }

    for (Value* prev : childrens_) {
        if (prev != nullptr) {
            prev->_backpropagation();
        } 
    }
}


Value sumManyValue(std::vector<Value*>& others) {
    // Sum all the data from the other values
    float total = 0.0f;
    std::vector<Value*> children;

    // Collect all the data and children references
    for (auto* val : others) {
        total += val->getData();
        children.push_back(val); // Add a pointer to each value
    }

    // Create a new Value object for the sum with the collected children
    Value result(total, children);

    // Set the backward function to distribute the gradient to all children
    result.setBackward([&result, children]() {
        float upperNodeGradient = result.getGrad();
        for (auto* child : children) {
            if (child) {
                child->setGrad(upperNodeGradient);
            }
        }
    });

    return result;
}