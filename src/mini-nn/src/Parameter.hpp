#pragma once
#include <memory>
#include "Value.hpp"

// A Parameter is a special type of Value that can be updated.
class Parameter : public Value {
public:
    // Constructor that calls the base class constructor
    Parameter(float data) : Value(data) {}

    // Static factory method for creating a Parameter
    static std::shared_ptr<Parameter> create(float data) {
        return std::make_shared<Parameter>(data);
    }

    // Method to update the data (e.g., during gradient descent)
    void updateData(float updateValue) { 
        data_ += updateValue; 
    }
};