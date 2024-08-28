#include "Value.hpp"

std::string Value::toString() const {
    std::string rpr = "<Value=" + std::to_string(data_) + ">";
    return rpr;
}

Value Value::applyOperator(const Value& other, std::function<float(float, float)> op_func, Op op) {
    // copy children of current instances
    std::vector<Value> children_copy = childrens_;

    // insert self and other into the list of children
    children_copy.push_back(other);
    children_copy.push_back(*this);

    // create a new value obj with new childrens
    return Value(op_func(data_, other.getData()), children_copy, op);
}

Value Value::operator+(const Value& other) {
    return applyOperator(other, [](float a, float b) { return a + b; }, Op::add);
}

Value Value::operator-(const Value& other) {
    return applyOperator(other, [](float a, float b) { return a - b; }, Op::sub);
}

Value Value::operator*(const Value& other) {
    return applyOperator(other, [](float a, float b) { return a * b; }, Op::mul);
}

Value Value::operator/(const Value& other) {
    if (other.getData() == 0.0) {
        std::__throw_runtime_error("Division by 0 impossible");
    }
    return applyOperator(other, [](float a, float b) { return a / b; }, Op::div);
}