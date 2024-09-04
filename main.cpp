#include <iostream>
#include "src/mini-nn/Values/Value.hpp"
#include "src/mini-nn/Activations/ActivationFunction.hpp"
#include "src/mini-nn/Activations/Static.hpp"
#include "src/mini-nn/Layers/Neuron.hpp"
#include "src/mini-nn/Layers/Dense.hpp"
#include "src/mini-nn/Tensor/Tensor.hpp"
#include "src/mini-nn/Operation/Gradient.hpp"
#include "src/mini-nn/Model/Sequential.hpp"
#include "src/mini-nn/Losses/Losses.hpp"

#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()


int main(){
    float stepSize = 5e-2; // i.e. lr
    float endloss = 1e-5;
    int num_data = 16*16;
    int input_data_sisze = 64;
    int num_epoch = 150;

    // seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_sisze, 32, Activations::Tanh));
    model.addLayer(Layers::Dense::create(32, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 1, Activations::Tanh));


    Tensor inputs = Tensor::randn({num_data, input_data_sisze});    // Init a full random tensor

    Tensor y({num_data});                                           // init a ytrue tensor (empty)

    for (int i = 0 ; i < num_data ; ++i) {
        y({i}) = Value::create((std::rand() % 2 == 0) ? -1 : 1);    // Fill it with rdn values
    }


    std::cout << "Start training " <<std::endl;
    for (int epoch = 1 ; epoch < num_epoch ; epoch++)
    {
        if (epoch % 5 == 0){
            stepSize = stepSize * 0.95;                             // lr decay
        }

        float loss = 0;                                             // Loss of the epoch ; only for user

        for(int i = 0 ; i < 16 ; ++i){                              // Iterate over all batches
            Tensor batchInput = inputs.slice(16*i, 16*i + 16);      // Get a batch for input
            Tensor batchTrue  = y.slice(16*i, 16*i + 16);           // Get a batch for ytrue

            Tensor x = model.forward(batchInput);                   // model forward
            auto fLoss = Losses::meanSquareError(x, batchTrue);     // Loss of the batch

            auto grad = Gradient::reverseTopologicalOrder(fLoss);   // Get the computational graph (backward)
            Gradient::backward(grad);                               // Perform backward propagation of gradient
            int nclip = Gradient::clipGrad(grad, 1.5f);             // Clip gradient

            model.update(stepSize);                                 // Update model weights (no optim yet)
            Gradient::zeroGrad(grad);                               // Reset all the grad (avoid accumulate over multiple batch)
            loss += fLoss->getData();                               // Increment epoch loss
        }
        std::cout << "Epoch : " << epoch << " ; Loss : " << loss / 16 << " ; lr : "<< stepSize << std::endl;
    }



    return 0;
}