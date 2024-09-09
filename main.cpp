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
    int bs = 4;
    int num_b = 1;
    int num_data = bs*num_b;
    int input_data_size = 784;
    int output_data_size = 1;
    int num_epoch = 100;

    // seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    auto model = Sequential();
    model.addLayer(Layers::Dense::create(input_data_size, 128, Activations::Tanh));
    model.addLayer(Layers::Dense::create(128, 64, Activations::Tanh));
    model.addLayer(Layers::Dense::create(64, 32, Activations::Tanh));
    model.addLayer(Layers::Dense::create(32, 16, Activations::Tanh));
    model.addLayer(Layers::Dense::create(16, output_data_size, Activations::Tanh));

    // set optim
    auto opt = Optimizers::SGD(model, stepSize);


    Tensor inputs = Tensor::randn({num_data, input_data_size});     // Init a full random tensor
    Tensor y({ num_data });                         // init a ytrue tensor (empty)


    for (int i = 0 ; i < y.size() ; ++i) {
        y({i}) = Value::create((std::rand() % 2 == 0) ? -1 : 1);
    }


    std::cout << "Start training " <<std::endl;
    for (int epoch = 1 ; epoch < num_epoch + 1 ; epoch++)
    {
        // if (epoch % 25 == 0){
        //     stepSize = stepSize * 0.5;                             // lr decay
        //     opt.setLearningRate(stepSize);
        // }

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
            // std::cout << "Gradients : " << std::flush;
            // auto grad = Gradient::reverseTopologicalOrder(fLoss);
            // Gradient::backward(grad);
            // for (auto& g : grad) {
            //     std::cout << g->getGrad() << " " << std::flush;
            // }
        }
        std::cout << std::endl;
    }

    return 0;
}

// int main() {
//     auto v1 = Value::create(1.0f);
//     auto v2 = Value::create(2.0f);
//     std::vector<std::shared_ptr<Value>> children = {v1, v2};
//     auto avgNode = Value::create(0.0f, std::make_unique<NodeOperation::Avg>());
//     avgNode->addChild(v1);
//     avgNode->addChild(v2);
//     avgNode->setGradient(1.0f);


//     avgNode->forward();
//     avgNode->backward();

//     return 0;
// }