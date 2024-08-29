#pragma once
#include "Value.hpp"

// A Parameter is a special value that can be updated.
class Parameter : public Value {
public:
    Parameter(float data) : Value(data) { }
    void updateData(float updateValue) { data_ += updateValue; }
};