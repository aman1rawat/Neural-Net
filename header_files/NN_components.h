#ifndef NN_COMPONENTS_H
#define NN_COMPONENTS_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "matrix.h"

typedef struct Layer{
	int size; // no of neurons in the layer
	Matrix * weighted_sum;
	Matrix * output;

	Matrix * weight; // weights for the layer
	Matrix * bias; //biases for the layer
	struct Layer * next_layer; // pointer to next layer (if any)
	struct Layer * prev_layer; // pointer to previous layer (if any)

	Matrix * error;
	char * activation;
}Layer;

typedef struct{
	int input_size;
	char *loss_function;
	Layer * layers; // head pointer to the linked list of layers
	Layer * tail_layer; // pointer to the last layer in the network
	double loss;
	double cost;
}NeuralNet;

Matrix* activate(Layer *layer);
Matrix* activationDerivative(Layer *layer);
Matrix* loss(NeuralNet *network, Matrix *output);
Matrix* lossDerivative(NeuralNet *network, Matrix *output);
// Matrix* cost();
// Matrix* costDerivative(Layer *tail_layer, Matrix *output);

#endif