#pragma once
#include <memory>
#include "Value.hpp"

// A Parameter is a special type of Value that can be updated.
enum NodeTypes {
    ANY, WEIGHT, BIAS, OUTPUT
};