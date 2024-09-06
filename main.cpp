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
#include "src/mini-nn/Optimizers/SGD.hpp"

#include <cstdlib> // For std::rand() and std::srand()
#include <ctime>   // For std::time()


int main(){
    float stepSize = 5e-3; // i.e. lr
    float endloss = 1e-5;
    int bs = 16;
    int num_b = 16;
    int num_data = bs*num_b;
    int input_data_size = 32;
    int output_data_size = 1;
    int num_epoch = 500;

    // seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_size, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, 8, Activations::Tanh));
    model.addLayer(Layers::Dense::create(8, 4, Activations::Tanh));
    model.addLayer(Layers::Dense::create(4, output_data_size, Activations::Tanh));

    // set optim
    auto opt = Optimizers::Adam(model, stepSize);


    Tensor inputs = Tensor::randn({num_data, input_data_size});     // Init a full random tensor
    // Tensor y({num_data, output_data_size});                         // init a ytrue tensor (empty)
    Tensor y({ num_data });                         // init a ytrue tensor (empty)

    for (int i = 0 ; i < num_data ; ++i) {
        // for (int j = 0 ; j < output_data_size ; ++j) {
        //     y({i, j}) = Value::create((std::rand() % 2 == 0) ? -1 : 1);
        // }
        y({ i }) = Value::create((std::rand() % 2 == 0) ? -1 : 1);
    }


    std::cout << "Start training " <<std::endl;
    for (int epoch = 1 ; epoch < num_epoch + 1 ; epoch++)
    {
        if (epoch % 50 == 0){
            stepSize = stepSize * 0.9;                             // lr decay
            opt.setLearningRate(stepSize);
        }

        float loss = 0;                                             // Loss of the epoch ; only for user

        for(int i = 0 ; i < num_b ; ++i){
            Tensor batchInput = inputs.slice(bs*i, bs*i + bs);      // Get a batch for input
            Tensor batchTrue  = y.slice(bs*i, bs*i + bs);           // Get a batch for ytrue

            Tensor x = model.forward(batchInput);                   // model forward
            std::shared_ptr<Value> fLoss = Losses::meanSquareError(x, batchTrue);   // Loss of the batch
            opt.update(fLoss);
            float iloss = fLoss->getData();
            loss += iloss;


                                          // Increment epoch loss
            std::cout << "Epoch : " << epoch << " ; batch : " << i + 1 << " / ";
            std::cout << num_b << " ; Loss : " << loss / (i + 1)<<std::flush;
            std::cout << " ; lr : " << opt.getLearningRate() <<std::flush;
            std::cout << "\r" <<std::flush;
            

            // std::cout << std::endl;
            // for(int b = 0; b < bs ; ++b){
            //     std::cout << "Value for batch " << i << "," << b << " : ";
            //     for (auto& val : batchInput[b].getValues()) {
            //         std::cout << val << " ";
            //     }
            //     std::cout << std::endl;
            // }            

            // std::cout << " ypred: ";
            // for (auto& val : x.getValues()) {
            //     std::cout << val << " ";
            // }
            // std::cout << " ; ytrue: ";
            // for (auto& val : batchTrue.getValues()) {
            //     std::cout << val << " ";
            // }
            // std::cout << "; " << std::endl; 
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


//     std::shared_ptr<Value> loss = Losses::binaryCrossEntropy(x, y);

//     return 0;
// }