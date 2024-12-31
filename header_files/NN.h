#ifndef NN_H
#define NN_H

#include "matrix.h"
#include "NN_components.h"

NeuralNet* buildNetwork(int input_size, char * loss_function);
void addLayer(NeuralNet *net, int size, char * activation);
void forwardPass(NeuralNet *network, Matrix *input);
void backwardPropagate(NeuralNet *network,Matrix *input, Matrix *output, double lr);
void trainNetwork(NeuralNet *net, Matrix *input, Matrix *output, double lr);
void printLayer(Layer *layer);

#endif