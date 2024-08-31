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

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(5, 5, Activations::Tanh));
    model.addLayer(Layers::Dense::create(5, 5, Activations::Tanh));
    model.addLayer(Layers::Dense::create(5, 5, Activations::Tanh));
    model.addLayer(Layers::Dense::create(5, 1, Activations::Sigmoid));
    
    
    std::vector<Tensor> inputs;
    for (int i = 0 ; i < input_size ; ++i) {
        inputs.push_back(Tensor::randn({ 5 }, 0, 1));
    }
    
    std::vector<int> y;
    for (int i = 0 ; i < input_size ; ++i) {
        int sum = 0; 
        for (auto val : inputs[i]) {
            sum += val->getData();
        }
        y.push_back(sum > 0 ? 1 : 0);
    }

    Tensor x;  
    std::cout << "Start training " <<std::endl;
    for (int j = 0 ; j < 500 ; ++j)
    {

        if (j%100 == 0 && j != 0){
            stepSize = stepSize*0.9;
        }

        Tensor outputs({input_size});

        for (int i = 0 ; i < input_size ; ++i) {
            x = model.forward(inputs[i]);

            auto loss = Math::pow(x({0})->sub(Value::create(y[i])), 2);
            outputs({i}) = loss;            
        }

        auto fLoss = Math::reduceSum(outputs);
        auto grad = Gradient::getGraphNodes(fLoss);
        Gradient::backward(grad);
        model.update(stepSize);
        Gradient::zeroGrad(grad);

        std::cout << "Iteration : " << j << " ; Loss : " << fLoss->getData() << " ; lr : "<< stepSize <<std::endl;   
    }
    
    
    return 0;
}
