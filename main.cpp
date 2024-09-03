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
    float stepSize = 5e-2; // i.e. lr
    int num_data = 8;
    int input_data_sisze = 64;

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_sisze, 32, Activations::Tanh));
    model.addLayer(Layers::Dense::create(32, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 1, Activations::Tanh));


    Tensor inputs = Tensor::randn({num_data, input_data_sisze});

    Tensor y({num_data});

    y({0}) = Value::create(1);
    y({1}) = Value::create(0);
    y({2}) = Value::create(-1);
    y({3}) = Value::create(1);
    y({4}) = Value::create(1);
    y({5}) = Value::create(0);
    y({6}) = Value::create(-1);
    y({7}) = Value::create(0);


    std::cout << "Start training " <<std::endl;
    int j = 0;
    float loss = 1;
    while (loss > 1e-2)
    {


        Tensor x = model.forward(inputs);

        Tensor sub = Math::ewSub(x, y);
        Tensor outputs = Math::pow(sub, 2);
        auto fLoss = Math::reduceMean(outputs);

        auto grad = Gradient::reverseTopologicalOrder(fLoss);
        Gradient::backward(grad);
        int nclip = Gradient::clipGrad(grad, 2.5f);
        model.update(stepSize);
        Gradient::zeroGrad(grad);

        loss = fLoss->getData();

        if (j%100 == 0 || loss <= 1e-2) {
            stepSize = stepSize*0.85;
            std::cout << "Iteration : " << j << " ; Loss : " << fLoss->getData() << " ; lr : "<< stepSize;
            std::cout << " ; nclip : " << nclip << std::endl;
            x.display();
            y.display();
        }
        j++;
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