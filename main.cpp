#include <iostream>
#include "src/mini-nn/src/Values/Value.hpp"
#include "src/mini-nn/src/Activations/ActivationFunction.hpp"
#include "src/mini-nn/src/Activations/Static.hpp"
#include "src/mini-nn/src/Layers/Neuron.hpp"
#include "src/mini-nn/src/Layers/Dense.hpp"

// int main(){
//     auto a = Value::create(2);
//     auto b = a->add(a);
//     // std::vector<Value*> inputs({&d, &f});
//     // Value L = sumManyValue(inputs);
//     // Value T = Activations::Tanh(L);

//     b->backward();
//     std::cout << "a : " <<a->toString() << std::endl;
//     std::cout << "b : " <<b->toString() << std::endl;
//     // std::cout << "T : " <<T->toString() << std::endl;
//     return 0;
// }

// int main() {
//     Value x1 = Value(2);
//     Value x2 = Value(0);

//     Value w1 = Value(-3);
//     Value w2 = Value(1);

//     Value b = Value(6.88137358);

//     Value x1w1 = x1*w1;
//     Value x2w2 = x2*w2;

//     Value s = x1w1 + x2w2;
//     Value sb = s + b;

//     Value out = Activations::Tanh(sb);

//     out.backward();

//     std::cout << "x1 : " <<x1.toString() << std::endl;
//     std::cout << "w1 : " <<w1.toString() << std::endl;
//     std::cout << "x2 : " <<x2.toString() << std::endl;
//     std::cout << "w2 : " <<w2.toString() << std::endl;
//     std::cout << "x2w2 : " <<x2w2.toString() << std::endl;
//     std::cout << "x1w1 : " <<x1w1.toString() << std::endl;
//     std::cout << "s : " <<s.toString() << std::endl;
//     std::cout << "sb : " <<sb.toString() << std::endl;
//     std::cout << "out : " <<out.toString() << std::endl;
//     return -1;
// }

// int main() {
//     Tanh tanh = Tanh();

//     Value a = Value(-2);
//     Value b = Value(3);

//     Value d = a * b;
//     Value e = a + b;
//     Value f = d * e;

//     f.backward();

//     std::cout << "a : " <<a.toString() << std::endl;
//     std::cout << "b : " <<b.toString() << std::endl;
//     std::cout << "d : " <<d.toString() << std::endl;
//     std::cout << "e : " <<e.toString() << std::endl;
//     std::cout << "f : " <<f.toString() << std::endl;
//     return -1;
// }

// int main() {
//     Value a(2.0); // Example value
//     Value b(3.0); // Example value

//     Value d = a * b;           // This works
//     Value e = a + b;           // This works
//     Value f = d * e;          // Now this will also work
//     Value out = Activations::Sigmoid(f);
//     // Value g = (a + b) * (a * b); // Chaining also works

//     // Perform backward pass if needed
//     out.backward();

//     std::cout << "out : " <<out.toString() << std::endl;
//     std::cout << "f : " <<f.toString() << std::endl;



//     return 0;
// }


int main(){
    Dense l1(5, 5);
    l1.func_ = &Activations::Tanh;
    Dense l2(5, 5);
    l2.func_ = &Activations::Tanh;
    Dense l3(5, 5);
    l3.func_ = &Activations::Tanh;
    Dense l4(5, 1);
    l4.func_ = &Activations::Tanh;

    std::vector<std::shared_ptr<Value>> inputs;
    for (int i = 0 ; i < 5 ; ++i){
        inputs.push_back(Value::create(1.0f));
    }

    std::vector<std::shared_ptr<Value>> x;
    

    x = l1.forward(inputs);
    x = l2.forward(x);
    x = l3.forward(x);
    x = l4.forward(x);

    l4.backward();

    std::cout << "Input Weight" << std::endl;
    for (auto val : inputs){
        std::cout << "inputs : " << val->toString() << std::endl;
    }


    for (auto val : l1.getParameters()){
        std::cout << "L1 Weight : " << val->toString() << std::endl;
    }
    std::cout << "L1 Biases" << std::endl;
    for (auto val : l1.getBiases()){
        std::cout << "L1 Biases : " << val->toString() << std::endl;
    }


    for (auto val : l2.getParameters()){
        std::cout << "l2 Weight : " << val->toString() << std::endl;
    }
    for (auto val : l2.getBiases()){
        std::cout << "l2 Biases : " << val->toString() << std::endl;
    }


    for (auto val : l3.getParameters()){
        std::cout << "l3 Weight : " << val->toString() << std::endl;
    }
    for (auto val : l3.getBiases()){
        std::cout << "l3 Biases : " << val->toString() << std::endl;
    }


    for (auto val : l4.getParameters()){
        std::cout << "l4 Weight : " << val->toString() << std::endl;
    }
    for (auto val : l4.getBiases()){
        std::cout << "l4 Biases : " << val->toString() << std::endl;
    }   
    std::cout << "out : " << x[0]->toString() << std::endl;
    
    return 0;
}

// int main() {
//     // Create two neurons, both with 5 inputs
//     Neuron neuron1(5);
//     // Neuron neuron2(1);  // The second neuron has 1 input, which is the output of neuron1

//     // Initialize inputs
//     std::vector<std::shared_ptr<Value>> inputs;
//     for (int i = 0; i < 5; ++i) {
//         inputs.push_back(Value::create(1.0)); // Initialize inputs to 1.0
//     }

//     // Forward pass through the first neuron
//     auto output1 = neuron1.forward(inputs);

//     // Use the output of the first neuron as the input to the second neuron
//     // std::vector<std::shared_ptr<Value>> inputs2;
//     // inputs2.push_back(output1);  // Connect the output of neuron1 to the input of neuron2

//     // Forward pass through the second neuron
//     // auto output2 = neuron2.forward(inputs2);

//     // Start backpropagation from the output of the second neuron
//     output1->backward();

//     // Display the gradients of the original inputs after backpropagation
//     for (int i = 0; i < inputs.size(); ++i) {
//         std::cout << "Input " << i << ": " << inputs[i]->toString() << std::endl;
//     }

//     // Display the gradients of the first neuron's weights and bias
//     auto weights1 = neuron1.getWeights();
//     for (int i = 0; i < weights1.size(); ++i) {
//         std::cout << "Neuron 1 Weight " << i << ": " << weights1[i]->toString() << std::endl;
//     }
//     std::cout << "Neuron 1 Bias: " << neuron1.getBias()->toString() << std::endl;
//     std::cout << "output: " << output1->toString() << std::endl;

//     // // Display the gradients of the second neuron's weights and bias
//     // auto weights2 = neuron2.getWeights();
//     // for (int i = 0; i < weights2.size(); ++i) {
//     //     std::cout << "Neuron 2 Weight " << i << ": " << weights2[i]->toString() << std::endl;
//     // }
//     // std::cout << "Neuron 2 Bias: " << neuron2.getBias()->toString() << std::endl;

//     return 0;
// }