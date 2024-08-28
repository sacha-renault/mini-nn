#include <iostream>
#include "src/mini-nn/src/Value.hpp"

int main(){
    Value a = Value(2);
    Value b = Value(-3);
    Value c = Value(10);
    std::cout << (a * b + c).toString() << std::endl;
    return -1;
}