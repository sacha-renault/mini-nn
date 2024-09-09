#pragma once
#include "Value.hpp"

std::shared_ptr<Value> operator+(std::shared_ptr<Value> left, std::shared_ptr<Value> right);
std::shared_ptr<Value> operator-(std::shared_ptr<Value> left, std::shared_ptr<Value> right);
std::shared_ptr<Value> operator*(std::shared_ptr<Value> left, std::shared_ptr<Value> right);
std::shared_ptr<Value> operator/(std::shared_ptr<Value> left, std::shared_ptr<Value> right);