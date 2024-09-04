#include <iostream>
#include "src/mini-nn/Values/Value.hpp"
#include "src/mini-nn/Activations/Activation.hpp"
#include "src/mini-nn/Layers/Neuron.hpp"
#include "src/mini-nn/Layers/Dense.hpp"
#include "src/mini-nn/Tensor/Tensor.hpp"
#include "src/mini-nn/Operation/Gradient.hpp"
#include "src/mini-nn/Model/Sequential.hpp"
#include "src/mini-nn/Losses/Losses.hpp"
#include "src/mini-nn/Optimizers/Adam.hpp"

#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()


int main(){
    float stepSize = 1e-2; // i.e. lr
    float endloss = 1e-5;
    int bs = 16;
    int num_b = 5;
    int num_data = bs*num_b;
    int input_data_size = 16;
    int num_epoch = 150;

    // seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_size, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 1, Activations::Tanh));

    // set optim
    auto opt = Optimizers::Adam(model, stepSize);


    Tensor inputs = Tensor::randn({num_data, input_data_size});    // Init a full random tensor
    Tensor y({num_data});                                           // init a ytrue tensor (empty)

    for (int i = 0 ; i < num_data ; ++i) {
        y({i}) = Value::create((std::rand() % 2 == 0) ? -1 : 1);    // Fill it with rdn values
    }


    std::cout << "Start training " <<std::endl;
    for (int epoch = 1 ; epoch < num_epoch + 1 ; epoch++)
    {
        if (epoch % 15 == 0){
            stepSize = stepSize * 0.9;                             // lr decay
            opt.setLearningRate(stepSize);
        }

        float loss = 0;                                             // Loss of the epoch ; only for user

        for(int i = 0 ; i < num_b ; ++i){
            Tensor batchInput = inputs.slice(bs*i, bs*i + bs);      // Get a batch for input
            Tensor batchTrue  = y.slice(bs*i, bs*i + bs);           // Get a batch for ytrue

            Tensor x = model.forward(batchInput);                   // model forward
            auto fLoss = Losses::meanSquareError(x, batchTrue);     // Loss of the batch
            opt.update(fLoss);
            float iloss = fLoss->getData();
            loss += iloss;                                          // Increment epoch loss
            std::cout << "Epoch : " << epoch << " ; batch : " << i + 1 << " / ";
            std::cout << num_b << " ; Loss : " << loss / (i + 1)<<std::flush;
            std::cout << " ; lr : " << opt.getLearningRate() <<std::flush;
            std::cout << "\r" <<std::flush;
        }
        std::cout << std::endl;
    }



    return 0;
}

// int main() {
//     int bs = 3;
//     Tensor x({ bs, 1 });
//     x({0, 0})->setValue(0.3);
//     x({1, 0})->setValue(0.6);
//     x({2, 0})->setValue(0.1);

//     Tensor y = Tensor::zeros({ bs, 1 });
//     y({0, 0})->setValue(1);


//     ValRef loss = Losses::binaryCrossEntropy(x, y);

//     return 0;
// }