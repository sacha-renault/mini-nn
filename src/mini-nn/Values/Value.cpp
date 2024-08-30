#include "Value.hpp"

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
    // Initialize the gradient of the highest node to 1.0
    grad_ = 1.0f;
    
    auto sorted_nodes = topologicalSort(shared_from_this());
    std::reverse(sorted_nodes.begin(), sorted_nodes.end());

    for (auto& node : sorted_nodes) {
        if (node->backward_) {
            node->backward_();
        }
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

/// @brief Get all the node sorted in topoligical order
/// @param root output node
/// @return unaccumulated gradient
std::vector<std::shared_ptr<Value>> topologicalSort(const std::shared_ptr<Value>& root) {
    std::vector<std::shared_ptr<Value>> sorted;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value>&)> dfs = [&](const std::shared_ptr<Value>& node) {
        if (visited.count(node)) return;
        visited.insert(node);
        for (const auto& child : node->getChildren()) {
            dfs(child);
        }
        sorted.push_back(node);  // Add node to the sorted list after visiting all children
    };

    dfs(root);
    return sorted;
}