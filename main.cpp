#include <iostream>
#include "src/mini-nn/Values/Value.hpp"
#include "src/mini-nn/Activations/ActivationFunction.hpp"
#include "src/mini-nn/Activations/Static.hpp"
#include "src/mini-nn/Layers/Neuron.hpp"
#include "src/mini-nn/Layers/Dense.hpp"
#include "src/mini-nn/Tensor/Tensor.hpp"
#include "src/mini-nn/Operation/Gradient.hpp"
#include "src/mini-nn/Model/Sequential.hpp"


int main(){
    float stepSize = 1e-2; // i.e. lr
    int input_size = 16;
    int input_data_sisze = 32;

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_sisze, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 4, Activations::Tanh));
    model.addLayer(Layers::Dense::create(4, 1, Activations::Tanh));
    
    
    std::vector<Tensor> inputs;
    for (int i = 0 ; i < input_size ; ++i) {
        inputs.push_back(Tensor::randn({ input_data_sisze }, -1, 1));
    }
    
    std::vector<int> y;
    for (int i = 0 ; i < input_size ; ++i) {
        int sum = 0; 
        for (auto val : inputs[i]) {
            sum += val->getData();
        }
        y.push_back(sum > 0 ? 1 : -1);
    }

    std::cout << "Start training " <<std::endl;
    for (int j = 0 ; j < 500 ; ++j)
    {
        if (j%100 == 0 && j != 0){
            stepSize = stepSize*0.85;
        }

        Tensor outputs({input_size});
        
        for (int i = 0 ; i < input_size ; ++i) {
            Tensor x = model.forward(inputs[i]);

            auto loss = Math::pow(x({0})->sub(Value::create(y[i])), 2);
            outputs({i}) = loss;            
        }

        // auto fLoss = Math::reduceSum(outputs);
        auto fLoss = outputs({0})->add(outputs({1}));
        auto grad = Gradient::getGraphNodes(fLoss);
        Gradient::backward(grad);
        model.update(stepSize);
        Gradient::derefGraph(grad);
        // std::cout << "Use count: " << x.data().use_count() << std::endl;

        std::cout << "Iteration : " << j << " ; Loss : " << fLoss->getData() << " ; lr : "<< stepSize <<std::endl;   
    }

    return 0;
}

// int main() {
//     Tensor tensor = Tensor::ones({3, 4, 5});

//     auto subtensor = Tensor(tensor.dim(), tensor.data());
//     subtensor({0,0,0}) = Value::create(0);

//     tensor.display();
//     return 0;
// }

// int main() {
//     Tensor tensor = Tensor::fromValue({20}, 1.0);

//    for (int i = 0 ; i < 20 ; ++i){
//     tensor({ i }) = Value::create((float)i);
//    }

//     tensor.display();
//     Tensor tensor = Tensor::ones({5, 5});
//     auto data = tensor.data();
//     std::cout << "Use count: " << tensor.data().use_count() << std::endl; // Should be 1 if no other references exist
//     std::cout << "Use count: " << data.use_count() << std::endl; // Should be 1 if no other references exist


//     return 0;
// }