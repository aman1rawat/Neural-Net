#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#include "NN.h"
#include "NN_components.h"

NeuralNet* buildNetwork(int input_size, char * loss_function){
    if(input_size<=0){
        printf("ERROR : invalid input size\n");
        return NULL;
    }
	NeuralNet *network = (NeuralNet*)malloc(sizeof(NeuralNet));
	network->input_size = input_size;
	network->layers = NULL;
	network->tail_layer = NULL;
    network->loss_function = (char*)malloc((strlen(loss_function)+1)*sizeof(char));
    strcpy(network->loss_function, loss_function);
	return network;
}

void addLayer(NeuralNet *network, int size, char *activation) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    layer->size = size;
    layer->weighted_sum = NULL;
    layer->output = NULL;
    layer->error = NULL;

    layer->bias = createMatrix(size, 1);

    if(!network->layers){
        network->layers = network->tail_layer = layer;
        layer->prev_layer = layer->next_layer = NULL;
        layer->weight = createMatrix(layer->size, network->input_size);
    } 
    else{
        layer->prev_layer = network->tail_layer; 
        layer->next_layer = NULL;
        network->tail_layer->next_layer = layer;   
        network->tail_layer = layer;
        layer->weight = createMatrix(layer->size, layer->prev_layer->size);
    }

    layer->activation = (char*)malloc((strlen(activation)+1)*sizeof(char));
    strcpy(layer->activation, activation);
    
    initializeMatrix(layer->weight);
    fillMatrix(layer->bias, 0);
}

void forwardPass(NeuralNet *network, Matrix *input){
    if(!network->layers){
        printf("ERROR : no layers in the network!\n");
        return;
    }

    for(Layer *current_layer = network->layers; current_layer; current_layer=current_layer->next_layer){
        if(current_layer->weighted_sum) freeMatrix(current_layer->weighted_sum);
        if(current_layer->output) freeMatrix(current_layer->output);

        if(current_layer==network->layers){
            current_layer->weighted_sum = dot(current_layer->weight, input);
        }
        else{
            current_layer->weighted_sum = dot(current_layer->weight, current_layer->prev_layer->output);
        }

        current_layer->weighted_sum = add(current_layer->weighted_sum, current_layer->bias);
        current_layer->output = activate(current_layer);
    }
}

void backwardPropagate(NeuralNet *network, Matrix *input, Matrix *output, double lr){
    //-------------------------------------first loop to get the error terms----------------------------------------------
    for(Layer *current_layer=network->tail_layer; current_layer; current_layer=current_layer->prev_layer){
        if(current_layer==network->tail_layer){
            current_layer->error = elementWiseMultiply(lossDerivative(network, output), activationDerivative(current_layer));
        }
        else{
            Matrix *propagatedError = dot(transpose(current_layer->next_layer->weight), current_layer->next_layer->error); 
            current_layer->error = elementWiseMultiply(propagatedError, activationDerivative(current_layer));
            freeMatrix(propagatedError);
        }
    }

    //------------------------------second loop to get the weight and bias gradients--------------------------------------
    for(Layer *current_layer=network->tail_layer; current_layer; current_layer=current_layer->prev_layer){
        Matrix *weight_gradient;
        if(current_layer==network->layers){
            weight_gradient = dot(current_layer->error, transpose(input));
        }
        else{
            weight_gradient = dot(current_layer->error, transpose(current_layer->prev_layer->output));
        }
        Matrix *bias_gradient = copyMatrix(current_layer->error);

        // clipGradients(weight_gradient, 1);
        // clipGradients(bias_gradient, 1);
        weight_gradient =  scale(weight_gradient, lr);
        bias_gradient = scale(bias_gradient, lr);
        current_layer->weight = subtract(current_layer->weight, weight_gradient);
        current_layer->bias = subtract(current_layer->bias, bias_gradient);

        freeMatrix(weight_gradient);
        freeMatrix(bias_gradient);
    }
}


void trainNetwork(NeuralNet *network, Matrix *input, Matrix *output, double lr) {
    if(!network || !network->layers){
        printf("ERROR in training the network\n");
        return;
    }

    forwardPass(network, input);
    network->loss = calculate_loss(network, output);
    backwardPropagate(network, input, output, lr);

    // printf("Network successfully trained!\n");
    return;
}

void printLayer(Layer *layer){
    printf("Weights:\n");
    printMatrix(layer->weight);
    printf("Bias:\n");
    printMatrix(layer->bias);
    printf("Weighted sum:\n");
    printMatrix(layer->weighted_sum);
    printf("Output:\n");
    printMatrix(layer->output);
    printf("---------------------------------\n");
}