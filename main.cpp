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


    Tensor inputs = Tensor::randn({input_size, input_data_sisze});

    Tensor y({input_size});
    for (int i = 0 ; i < input_size ; ++i) {
        int sum = 0;
        for (auto& val : inputs[i]) {
            sum += val->getData();
        }
        y({i}) = Value::create(sum > 0 ? 1 : -1);
    }

    std::cout << "Start training " <<std::endl;
    int j = 0;
    float loss = 1;
    while (loss > 1e-2)
    {
        j++;
        if (j%100 == 0 && j != 0){
            stepSize = stepSize*0.85;
        }

        Tensor outputs({input_size});

        for (int i = 0 ; i < input_size ; ++i) {
            auto in = inputs[i];
            Tensor x = model.forward(in);

            auto loss = Math::pow(x({0})->sub(y({i})), 2);
            outputs({i}) = loss;
        }

        // auto fLoss = Math::reduceSum(outputs);
        auto fLoss = Math::reduceMean(outputs);
        auto grad = Gradient::reverseTopologicalOrder(fLoss);
        Gradient::backward(grad);
        int nclip = Gradient::clipGrad(grad);
        model.update(stepSize);
        Gradient::derefGraph(grad);

        std::cout << "Iteration : " << j << " ; Loss : " << fLoss->getData() << " ; lr : "<< stepSize;
        std::cout << " ; nclip : " << nclip << std::endl;
        loss = fLoss->getData();
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
//     Tensor tensor = Tensor::zeros({20});

//     tensor.reshape({4, 5});

//     auto subtensor = tensor[0];

//     int size = subtensor.size();
//     std::cout << "";
//     for (auto& d : subtensor){
//         d->setValue(1.0);
//     }

//     tensor.display();
//     subtensor.display();

//     return 0;
// }