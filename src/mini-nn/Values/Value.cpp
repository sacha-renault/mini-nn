#include "Value.hpp"
#include "../Operation/Gradient.hpp"

std::string Value::toString() const {
    std::string rpr = "<Value=" + std::to_string(data_); // data value
    rpr += " ; Grad=" +  std::to_string(grad_); // grad value
    rpr += ">";
    return rpr;
}

void Value::backward() {
    if (operation_) {
        operation_->backward(children_, shared_from_this());
    }
}

void Value::forward() {
    if (operation_){
        operation_->forward(children_, shared_from_this());
    }
}

void Value::derefGraph() {
    zeroGrad();
    children_.clear();  // clean all refs to child, since it's share_ptr,
                        // any shared_ptr that doesn't have owner will be free
}

void Value::zeroGrad() {
    grad_ = 0.0f;
}