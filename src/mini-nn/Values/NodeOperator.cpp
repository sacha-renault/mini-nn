#include "NodeOperator.hpp"
#include "NodeOperation.hpp"

std::shared_ptr<Value> operator+(std::shared_ptr<Value> left, std::shared_ptr<Value> right) {
    // calculate result
    float res = left->getData() + right->getData();

    // create the out node
    auto outNode = Value::create(res, std::make_unique<NodeOperation::Add>());

    // add children into outNode
    outNode->addChild(left);
    outNode->addChild(right);

    // return the output
    return outNode;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> left, std::shared_ptr<Value> right) {
    // calculate result
    float res = left->getData() - right->getData();

    // create the out node
    auto outNode = Value::create(res, std::make_unique<NodeOperation::Sub>());

    // add children into outNode
    outNode->addChild(left);
    outNode->addChild(right);

    // return the output
    return outNode;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> left, std::shared_ptr<Value> right) {
    // calculate result
    float res = left->getData() * right->getData();

    // create the out node
    auto outNode = Value::create(res, std::make_unique<NodeOperation::Mul>());

    // add children into outNode
    outNode->addChild(left);
    outNode->addChild(right);

    // return the output
    return outNode;
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> left, std::shared_ptr<Value> right) {
    // calculate result
    float res = left->getData() / right->getData();

    // create the out node
    auto outNode = Value::create(res, std::make_unique<NodeOperation::Div>());

    // add children into outNode
    outNode->addChild(left);
    outNode->addChild(right);

    // return the output
    return outNode;
}