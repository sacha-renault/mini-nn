#include "NodeOperation.hpp"
#include "Value.hpp"  // Include the full definition of Value

namespace NodeOperation {

    // Add Implementation
    void Add::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float sum = 0;
        for(auto& child : children) {
            sum += child->getData();
        }
        output->setValue(sum);
    }

    void Add::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float parentGrad = output->getGrad();
        for(const auto& child : children) {
            child->accumulateGrad(parentGrad);
        }
    }




    // Avg Implementation
    void Avg::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float sum = 0;
        for(auto& child : children) {
            sum += child->getData();
        }
        output->setValue(sum / children.size());
    }

    void Avg::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float parentGrad = output->getGrad() / children.size();
        for(const auto& child : children) {
            child->accumulateGrad(parentGrad);
        }
    }




    // Mul Implementation
    void Mul::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        if (children.size() != 2) {
            throw std::runtime_error("operation Mul must have exactly two children");
        }
        output->setValue(children[0]->getData() * children[1]->getData());
    }

    void Mul::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float parentGrad = output->getGrad();
        children[0]->accumulateGrad(parentGrad * children[1]->getData());
        children[1]->accumulateGrad(parentGrad * children[0]->getData());
    }




    // Sub Implementation
    void Sub::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        if (children.size() != 2) {
            throw std::runtime_error("operation Sub must have exactly two children");
        }
        output->setValue(children[0]->getData() - children[1]->getData());
    }

    void Sub::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float parentGrad = output->getGrad();
        children[0]->accumulateGrad(parentGrad);
        children[1]->accumulateGrad(-parentGrad);

        // out->addBackward([out, other, this]() {
        //     other->accumulateGrad(-out->getGrad()); // Negative gradient for subtraction
        //     this->accumulateGrad(out->getGrad());
        // });
    }




    // Div implem
    void Div::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        float parentGrad = output->getGrad();
        float x = children[0]->getData();
        float y = children[1]->getData();
        children[0]->accumulateGrad(parentGrad / y);
        children[1]->accumulateGrad(-parentGrad * x / (y * y));
    }

    void Div::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        if (children.size() != 2) {
            throw std::runtime_error("operation Div must have exactly two children");
        }
        output->setValue(children[0]->getData() / children[1]->getData());
    }




    // Function1 Implementation
    Function1::Function1(std::function<float(float)> forward, std::function<float(float)> backward)
        : forward_(forward), backward_(backward) { }

    void Function1::forward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        if (children.size() != 1) {
            throw std::runtime_error("operation Function1 must have exactly 1 child");
        }
        output->setValue(forward_(children[0]->getData()));
    }

    void Function1::backward(const std::vector<std::shared_ptr<Value>>& children, std::shared_ptr<Value> output) {
        if (children.size() != 1) {
            throw std::runtime_error("operation Function1 must have exactly 1 child");
        }
        float localGradient = backward_(children[0]->getData());
        children[0]->accumulateGrad(output->getGrad() * localGradient);
    }

} // namespace NodeOperation
